// Deterministic examples indexed by a shared database counter.
const SUBJECTS=[
 ['astronomy','planets','Planets orbit stars because of gravity.'],
 ['biology','cells','Cells are basic units of life.'],
 ['geography','oceans',"Oceans cover most of Earth's surface."],
 ['physics','light','Light travels through a vacuum.'],
 ['chemistry','water','Water is made from hydrogen and oxygen.'],
 ['computing','bits','A bit is either zero or one.'],
 ['grammar','sentences','A sentence normally expresses a complete thought.'],
 ['history','evidence','Historians compare sources to study the past.'],
 ['weather','clouds','Clouds contain tiny droplets or ice crystals.'],
 ['music','rhythm','Rhythm describes the timing of musical events.'],
 ['ecology','habitats','A habitat is where an organism lives.'],
 ['math','fractions','A fraction represents part of a whole.'],
 ['engineering','testing','Engineers test designs to find problems.'],
 ['language','translation','Translation conveys meaning between languages.']
];
const GREETS=['hello','hi','hey','good morning','good evening','hello there','hi AXON AI','hey there'];
const RESPONSES=['Hello! How can I help?','Hi! What would you like to know?','Hey! What are you working on?','Hello there. What question do you have?','Hi! Tell me what you would like to explore.'];
const QUESTIONS=['What can you tell me about','Explain','Tell me about','What does science say about','Give a short explanation of'];
const CHAT=[
 ['What can you do?','I can help with basic questions and calculations. I am still learning.'],
 ['What is an AI model?','An AI model learns patterns from data and uses them to make predictions.'],
 ['How do computers learn?','A training algorithm adjusts model weights using feedback from examples.'],
 ['What is a token?','A token is a small unit of text represented by a numerical ID.'],
 ['How do you get better at answers?','My training system updates model weights and checks the results.'],
 ['What should I do when I am unsure?','Check reliable information rather than guessing.'],
 ['Can you make mistakes?','Yes. Verify important answers with reliable sources.'],
 ['How can I start a project?','Choose a specific goal, build a small prototype, and test it.'],
 ['What is debugging?','Debugging means finding and correcting problems in a program.']
];
const NAMES=['Mira','Alex','Sam','Taylor','Jordan','Jamie','Robin','Casey','Dana','Riley','Morgan','Avery'];
const THINGS=['apples','books','pens','stones','coins','tiles','cups','tickets','cards','boxes'];
function mix(index){
 let x=(index^0x9e3779b9)>>>0;
 x=Math.imul(x^(x>>>16),0x7feb352d)>>>0;
 x=Math.imul(x^(x>>>15),0x846ca68b)>>>0;
 return (x^(x>>>16))>>>0;
}
export function curriculumExample(index){
 if(!Number.isSafeInteger(index)||index<0||index>=1_000_000_000)throw Error('Invalid corpus index');
 const rng=mix(index),family=index%14,seed=Math.floor(index/14),k=seed+2;
 const name=NAMES[(rng>>>7)%NAMES.length],thing=THINGS[rng%THINGS.length];
 let concept,question,response,a,b;
 switch(family){
  case 0:
   concept='greetings';question=GREETS[rng%GREETS.length]+' '+name+' '+seed;
   response=RESPONSES[(rng>>>9)%RESPONSES.length];break;
  case 1:
   concept='addition';a=k%500000;b=(k*13+7)%400000;
   question='What is '+a+' plus '+b+'?';response=String(a+b);break;
  case 2:
   concept='subtraction';a=k%600000;b=(k*17+3)%200000;
   question='Calculate '+(a+b)+' minus '+b+'.';response=String(a);break;
  case 3:
   concept='multiplication';a=k%50000;b=2+(k%21);
   question='What is '+a+' multiplied by '+b+'?';response=String(a*b);break;
  case 4:
   concept='division';a=1+(k%75000);b=2+(k%31);
   question='Calculate '+(a*b)+' divided by '+b+'.';response=String(a);break;
  case 5:
   concept='comparison';a=k%200000;b=a+1+(k%500);
   question='Which is larger, '+a+' or '+b+'?';response=String(b);break;
  case 6:
   concept='parity';a=k%1000000;
   question='Is '+a+' even or odd?';response=a%2?'odd':'even';break;
  case 7:
   concept='unit conversion';a=k%100000;
   question='How many centimeters are in '+a+' meters?';response=String(a*100);break;
  case 8:
   concept='square';a=k%1000;
   question='What is '+a+' squared?';response=String(a*a);break;
  case 9:
   concept='binary';a=k%65536;
   question='Write '+a+' in binary.';response=a.toString(2);break;
  case 10:
   concept='worded arithmetic';a=k%60;b=(k*7)%60;
   question=name+' has '+a+' '+thing+' and gets '+b+' more. How many '+thing+' now?';
   response=(a+b)+' '+thing+'.';break;
  case 11:{
   const t=SUBJECTS[rng%SUBJECTS.length];concept=t[0];
   question=QUESTIONS[(rng>>>10)%QUESTIONS.length]+' '+t[1]+' (example '+seed+').';
   response=t[2];break;
  }
  case 12:{
   const t=CHAT[rng%CHAT.length];concept='conversation';
   question=t[0]+' '+(seed%4===0?name+' is asking.':'example '+seed);
   response=t[1];break;
  }
  default:
   concept='sequencing';a=k%200000;
   question='Continue: '+a+', '+(a+2)+', '+(a+4)+'. What comes next?';
   response=String(a+6);
 }
 return {id:index,concept,messages:[{role:'user',content:question}],response};
}
export function corpusLine(index){
 const e=curriculumExample(index);
 return 'User: '+e.messages[0].content+'\tAssistant: '+e.response+'\n';
}
