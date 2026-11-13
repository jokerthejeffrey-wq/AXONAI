using System;
using System.IO;
using System.IO.Compression;
using System.Net;
using System.Reflection;
using System.Diagnostics;
using System.Security.Cryptography;
using System.Threading;
using Microsoft.Win32;
class Launcher {
 const string Device="__DEVICE__";
 const string InstallerHash="__INSTALLER_HASH__";
 const string PayloadHash="__PAYLOAD_HASH__";
 static string Quote(string s){return "\""+s.Replace("\"","\\\"")+"\"";}
 static int Run(string exe,string args,string directory){
  var p=Process.Start(new ProcessStartInfo(exe,args){UseShellExecute=false,WorkingDirectory=directory});
  p.WaitForExit();return p.ExitCode;
 }
 static string Hash(byte[] bytes){using(var sha=SHA256.Create())return BitConverter.ToString(sha.ComputeHash(bytes)).Replace("-","").ToLowerInvariant();}
 static bool Healthy(string path){
  try{return File.Exists(path)&&Run(path,"-c \"import sys,struct; assert sys.version_info[:2]==(3,12) and struct.calcsize('P')==8; import venv,ensurepip\"",Path.GetDirectoryName(path))==0;}catch{return false;}
 }
 static string Python(string home){
  string local=Path.Combine(home,"python312","python.exe");if(Healthy(local))return local;
  foreach(var hive in new[]{Registry.CurrentUser,Registry.LocalMachine}){
   using(var key=hive.OpenSubKey(@"Software\Python\PythonCore\3.12\InstallPath")){
    var dir=key==null?null:key.GetValue("") as string;
    if(dir!=null&&Healthy(Path.Combine(dir,"python.exe")))return Path.Combine(dir,"python.exe");
   }
  }
  Console.WriteLine("Downloading Python 3.12 for AXONAI...");
  string installer=Path.Combine(home,"python-3.12.10-amd64.exe");
  ServicePointManager.SecurityProtocol=SecurityProtocolType.Tls12;
  using(var web=new WebClient())web.DownloadFile("https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe",installer);
  if(Hash(File.ReadAllBytes(installer))!=InstallerHash)throw new Exception("Python download checksum failed. Run again to retry.");
  Console.WriteLine("Installing Python for this Windows account...");
  int code=Run(installer,"/quiet InstallAllUsers=0 Include_launcher=0 Include_test=0 Include_doc=0 Include_tcltk=0 PrependPath=0 Shortcuts=0 TargetDir="+Quote(Path.Combine(home,"python312")),home);
  if(code!=0&&code!=3010)throw new Exception("Python installer returned "+code);
  File.Delete(installer);
  if(!Healthy(local))throw new Exception("Python setup needs attention. Install Python 3.12 x64 from python.org and retry.");
  return local;
 }
 static string Extract(string home){
  string folder=Path.Combine(home,"app-"+PayloadHash.Substring(0,16));
  Directory.CreateDirectory(folder);
  // Validate and restore the bundled bytecode on each launch. No source ZIP download.
  using(var stream=Assembly.GetExecutingAssembly().GetManifestResourceStream("payload.zip")){
   using(var memory=new MemoryStream()){
    stream.CopyTo(memory);var bytes=memory.ToArray();if(Hash(bytes)!=PayloadHash)throw new Exception("Embedded application checksum failed.");
    using(var zip=new ZipArchive(new MemoryStream(bytes),ZipArchiveMode.Read)){
     foreach(var entry in zip.Entries){
      string target=Path.GetFullPath(Path.Combine(folder,entry.FullName));
      if(!target.StartsWith(folder+Path.DirectorySeparatorChar,StringComparison.OrdinalIgnoreCase))throw new Exception("Invalid embedded path.");
      Directory.CreateDirectory(Path.GetDirectoryName(target));
      using(var input=entry.Open())using(var output=File.Create(target))input.CopyTo(output);
     }
    }
   }
  }
  return folder;
 }
 static int Main(string[] args){
  bool test=Array.IndexOf(args,"--self-test")>=0,setupOnly=Array.IndexOf(args,"--setup-only")>=0;
  try{
   Console.Title="AXONAI Trainer ["+Device+"]";
   Console.WriteLine("AXONAI Trainer ["+Device+"]\n");
   if(!Environment.Is64BitOperatingSystem)throw new Exception("64-bit Windows is required.");
   string home=Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),"AXONAI");
   Directory.CreateDirectory(home);string folder,python=null;
   using(var mutex=new Mutex(false,@"Local\AXONAI-Setup")){
    Console.WriteLine("Preparing trainer...");
    try{mutex.WaitOne();}catch(AbandonedMutexException){}
    try{folder=Extract(home);if(!test)python=Python(home);}finally{mutex.ReleaseMutex();}
   }
   if(test){
    if(!File.Exists(Path.Combine(folder,"trainer","run.pyc"))||!File.Exists(Path.Combine(folder,"student.pyc")))throw new Exception("Trainer payload missing.");
    foreach(var path in Directory.GetFiles(folder,"*.py",SearchOption.AllDirectories))throw new Exception("Unexpected plain source: "+path);
    Console.WriteLine("EXE payload verified: "+Device+" conversational trainer.");return 0;
   }
   if(setupOnly){Console.WriteLine("Python and trainer setup verified.");return 0;}
   Console.WriteLine("First setup downloads dependencies, conversation data and the local teacher.\nCtrl+C stops the program.\n");
   string cli=Quote(Path.Combine(folder,"trainer","start.pyc"))+(Device=="GPU"?" --gpu":"");
   foreach(string arg in args)cli+=" "+Quote(arg);
   int result=Run(python,cli,folder);
   Console.WriteLine("\nTrainer finished. Press Enter to close.");Console.ReadLine();return result;
  }catch(Exception e){Console.Error.WriteLine("\nAXONAI: "+e.Message);if(!test&&!setupOnly){Console.WriteLine("Press Enter to close.");Console.ReadLine();}return 1;}
 }
}
