import {chromium} from 'playwright';
import assert from 'node:assert/strict';
const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
const widths=[320,360,390,428,768,1024];
const pages=[
 {file:'index.html',checks:['header nav','.hero','.traincols','.progress-wrap','.feed']},
 {file:'chat.html',checks:['.chat-header','#thread','#composer','.composer-area']},
 {file:'downloads.html',checks:['main','.items','.item']},
 {file:'docs.html',checks:['main']}
];
try{
  for(const {file,checks} of pages){
    for(const width of widths){
      const page=await browser.newPage({viewport:{width,height:800},isMobile:true,hasTouch:true,deviceScaleFactor:1});
      await page.goto('http://127.0.0.1:8765/'+file,{waitUntil:'domcontentloaded',timeout:30000});
      const info=await page.evaluate((selectors)=>{
        const data={width:innerWidth,scrollWidth:document.documentElement.scrollWidth,offenders:[]};
        for(const selector of selectors){
          for(const el of document.querySelectorAll(selector)){
            const r=el.getBoundingClientRect();
            if(r.width && (r.left < -2 || r.right>innerWidth+2)){
              data.offenders.push({selector,left:Math.round(r.left),right:Math.round(r.right)});
            }
          }
        }
        if(document.querySelector('#composer')){
          const input=document.querySelector('#prompt');
          data.promptFontSize=Number.parseFloat(getComputedStyle(input).fontSize);
          data.sendSize=Math.round(document.querySelector('#send').getBoundingClientRect().width);
        }
        return data;
      },checks);
      console.log(file,width,'scrollWidth',info.scrollWidth,'overflow',JSON.stringify(info.offenders));
      assert.ok(info.scrollWidth<=info.width+2,file+' overflows viewport '+width+': '+JSON.stringify(info));
      assert.deepEqual(info.offenders,[],file+' has clipped important elements at '+width);
      if(file==='chat.html'&&width<=390){
        assert.ok(info.promptFontSize>=16,'iOS input zoom risk');
        assert.ok(info.sendSize>=40,'Send button too small');
      }
      await page.close();
    }
  }
  console.log('Responsive viewport tests passed.');
}finally{await browser.close();}
