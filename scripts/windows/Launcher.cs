using System;
using System.IO;
using System.IO.Compression;
using System.Net;
using System.Reflection;
using System.Diagnostics;
using System.Security.Cryptography;
using System.Threading;
using System.Runtime.InteropServices;
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
 static string TimeLeft(long done,long total,DateTime start){
  if(done<=0||total<=0)return "calculating";
  double elapsed=Math.Max(1,(DateTime.UtcNow-start).TotalSeconds);
  double remaining=Math.Max(0,(total-done)*elapsed/done);
  if(remaining>=3600)return "~"+(int)(remaining/3600)+"h "+(int)((remaining%3600)/60)+"m";
  if(remaining>=60)return "~"+(int)(remaining/60)+"m "+(int)(remaining%60)+"s";
  return "~"+(int)remaining+"s";
 }
 static void ShowProgress(string label,int percent,string eta){
  int filled=Math.Max(0,Math.Min(20,percent/5));
  var bar=new string('=',filled)+new string(' ',20-filled);
  var line=label+" ["+bar+"] "+percent+"%  remaining: "+eta;
  Console.Write("\r"+line.PadRight(95));
 }
 static void DownloadPython(string url,string path){
  ServicePointManager.SecurityProtocol=SecurityProtocolType.Tls12;
  using(var web=new WebClient())
   using(var input=web.OpenRead(url))
    using(var output=new FileStream(path,FileMode.Create,FileAccess.Write,FileShare.None)){
     long total=web.ResponseHeaders==null?-1:web.ResponseHeaders["Content-Length"]==null?-1:long.Parse(web.ResponseHeaders["Content-Length"]);
     var start=DateTime.UtcNow;var last=start;
     var buffer=new byte[131072];long read=0;int n;
     ShowProgress("Python download",0,"calculating");
     while((n=input.Read(buffer,0,buffer.Length))>0){
      output.Write(buffer,0,n);read+=n;
      if((DateTime.UtcNow-last).TotalMilliseconds>=500){
       var percent=total>0?Math.Min(99,(int)(100*read/total)):0;
       ShowProgress("Python download",percent,TimeLeft(read,total,start));
       last=DateTime.UtcNow;
      }
     }
     if(total>0&&read!=total)throw new Exception("Python download incomplete.");
     ShowProgress("Python download",100,"~0s");Console.WriteLine();
    }
 }
 static int InstallPython(string installer,string args,string home){
  using(var p=Process.Start(new ProcessStartInfo(installer,args){UseShellExecute=false,WorkingDirectory=home})){
   var start=DateTime.UtcNow;
   while(!p.WaitForExit(1000)){
    // Installer does not expose completion bytes, so this is an approximate stage ETA.
    var elapsed=(DateTime.UtcNow-start).TotalSeconds;
    int estimate=(int)Math.Min(95,elapsed/90*100);
    ShowProgress("Python setup",estimate,elapsed<4?"calculating":"estimated");
   }
   ShowProgress("Python setup",100,"~0s");Console.WriteLine();
   return p.ExitCode;
  }
 }

 static string Python(string home){
  string local=Path.Combine(home,"python312","python.exe");if(Healthy(local))return local;
  foreach(var hive in new[]{Registry.CurrentUser,Registry.LocalMachine}){
   using(var key=hive.OpenSubKey(@"Software\Python\PythonCore\3.12\InstallPath")){
    var dir=key==null?null:key.GetValue("") as string;
    if(dir!=null&&Healthy(Path.Combine(dir,"python.exe")))return Path.Combine(dir,"python.exe");
   }
  }
  string installer=Path.Combine(home,"python-3.12.10-amd64.exe");
  if(!File.Exists(installer)||Hash(File.ReadAllBytes(installer))!=InstallerHash)
   DownloadPython("https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe",installer);
  if(Hash(File.ReadAllBytes(installer))!=InstallerHash)throw new Exception("Python download checksum failed. Run again to retry.");
  int code=InstallPython(installer,"/quiet InstallAllUsers=0 Include_launcher=0 Include_test=0 Include_doc=0 Include_tcltk=0 PrependPath=0 Shortcuts=0 TargetDir="+Quote(Path.Combine(home,"python312")),home);
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
 // A mouse selection in the classic Windows console can pause a program's output.
 [DllImport("kernel32.dll")] static extern IntPtr GetStdHandle(int handle);
 [DllImport("kernel32.dll")] static extern bool GetConsoleMode(IntPtr handle,out uint mode);
 [DllImport("kernel32.dll")] static extern bool SetConsoleMode(IntPtr handle,uint mode);
 static void DisableQuickEdit(){
  const int StandardInput=-10;
  const uint QuickEdit=0x0040,ExtendedFlags=0x0080;
  var input=GetStdHandle(StandardInput);
  uint mode;
  if(GetConsoleMode(input,out mode))SetConsoleMode(input,(mode|ExtendedFlags)&~QuickEdit);
 }
 static int Main(string[] args){
  bool test=Array.IndexOf(args,"--self-test")>=0,setupOnly=Array.IndexOf(args,"--setup-only")>=0;
  bool installOnly=Array.IndexOf(args,"--install-only")>=0;
  try{
   DisableQuickEdit();
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
    if(!File.Exists(Path.Combine(folder,"trainer","run.pyc"))||!File.Exists(Path.Combine(folder,"trainer","numpy_student.pyc")))throw new Exception("Trainer payload missing.");
    foreach(var path in Directory.GetFiles(folder,"*.py",SearchOption.AllDirectories))throw new Exception("Unexpected plain source: "+path);
    Console.WriteLine("EXE payload verified: "+Device+" conversational trainer.");return 0;
   }
   if(setupOnly){Console.WriteLine("Python and trainer setup verified.");return 0;}
   Console.WriteLine("First setup downloads dependencies, conversation data.\nCtrl+C stops the program.\n");
   string cli=Quote(Path.Combine(folder,"trainer","start.pyc"))+(Device=="GPU"?" --gpu":"");
   foreach(string arg in args)cli+=" "+Quote(arg);
   int result=Run(python,cli,folder);
   if(!installOnly){Console.WriteLine("\nTrainer finished. Press Enter to close.");Console.ReadLine();}
   return result;
  }catch(Exception e){Console.Error.WriteLine("\nAXONAI: "+e.Message);if(!test&&!setupOnly&&!installOnly){Console.WriteLine("Press Enter to close.");Console.ReadLine();}return 1;}
 }
}
