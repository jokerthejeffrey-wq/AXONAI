// Only solve strictly matched two-integer arithmetic queries. No eval or dynamic code execution.
const integer='(-?\\d{1,10})';
export function simpleArithmetic(text){
  const s=String(text||'').trim().replace(/[?!.]+$/,'').replace(/[“”"']/g,'').trim();
  let m=s.match(/^(-?\d{1,10})\s*([+\-*×÷/])\s*(-?\d{1,10})$/);
  if(!m) m=s.match(/^(?:what is|calculate|compute|solve|find|tell me)\s+(-?\d{1,10})\s*(minus|subtract|plus|added to|times|multiplied by|divided by|[-+*×÷/])\s*(-?\d{1,10})$/i);
  if(!m){
    const sub=s.match(/^(?:what is|calculate|compute|solve)\s+(-?\d{1,10})\s*(?:minus|subtract|-)\s*(-?\d{1,10})(?:\s*(?:equal to|equals))?$/i);
    if(sub)m=[sub[0],sub[1],'-',sub[2]];
  }
  if(!m)return null;
  const a=Number(m[1]),b=Number(m[3]);
  if(!Number.isSafeInteger(a)||!Number.isSafeInteger(b))return null;
  let n;switch(m[2].toLowerCase()){
    case '+':case 'plus':case 'added to':n=a+b;break;
    case '-':case 'minus':case 'subtract':n=a-b;break;
    case '*':case 'times':case 'multiplied by':case '×':n=a*b;break;
    case '/':case '÷':case 'divided by':n=b===0?null:a/b;break;
    default:return null;
  }
  if(n===null)return 'Division by zero is undefined.';
  return Number.isFinite(n)&&Math.abs(n)<1e16?String(n):null;
}
