const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const C=require('./dist/replay-core.js');
const D=JSON.parse(fs.readFileSync(path.join(__dirname,'dist/replay-data.json'),'utf8'));
const F=JSON.parse(fs.readFileSync(path.join(__dirname,'dist/july-forecasts.json'),'utf8'));
assert.equal(D.originCount,738);
assert.equal(F.cases,738*262*6);
assert.equal(F.zones.length,262);
assert.equal(F.publicationDelayAfterHourEnd,2);
assert.equal(F.firstHistoryHours,165);
assert.equal(F.modelSha256,'af5e4c75c4b9bb11e0783ad3223955f268fef194efcef10db7b14ec2116124a5');
const ids=new Set();
for(const z of F.zones){
 assert(z.id>=2&&z.id<=263&&!ids.has(z.id));ids.add(z.id);
 assert.equal(z.original.length,738*6);assert.equal(z.correction.length,738);assert.equal(z.historyCount.length,738);
 assert(z.original.every(v=>Number.isFinite(v)&&v>=0));assert(z.correction.every(Number.isFinite));
 assert(z.historyCount.every(v=>Number.isInteger(v)&&v>0&&v<=168));
 for(let i=0;i<738;i++)for(let h=1;h<=6;h++){
  assert.equal(C.adjustedForecast(z,i,h,0),z.original[i*6+h-1]);
  for(const alpha of [.25,1])assert.equal(C.adjustedForecast(z,i,h,alpha),Math.max(0,z.original[i*6+h-1]+alpha*z.correction[i]));
 }
}
assert(!ids.has(1));
const first=JSON.parse(fs.readFileSync(path.join(__dirname,'dist/first-origin-forecasts.json'),'utf8'));
const zone186=F.zones.find(z=>z.id===186);
for(const [z,h,p]of first.rows)if(z===186)assert(Math.abs(zone186.original[h-1]-p)<.001);
for(const h of [1,2,3,4,5,6]){
 assert(!C.isRevealed(h,h+2));assert(C.isRevealed(h,h+3));
}
for(const index of [0,300,737])for(const elapsed of [0,2,3,8,9]){
 const rows=C.sixHours(D,186,index,elapsed,new Map());
 assert.equal(rows.length,6);
 for(const row of rows)assert.equal(row.actual===null,elapsed<row.horizon+3);
}
const header='forecast_origin_utc,location_id,forecast_horizon,prediction\n';
assert.equal(C.importForecasts(header+'2026-07-01T04:00:00Z,186,1,100.5\n',D).get('0:186:1'),100.5);
for(const line of ['2026-07-01T04:00:00Z,1,1,2','2026-07-01T04:00:00Z,186,1,-1','2026-08-01T04:00:00Z,186,1,2'])assert.throws(()=>C.importForecasts(header+line+'\n',D));
const html=fs.readFileSync(path.join(__dirname,'dist/index.html'),'utf8');
const app=fs.readFileSync(path.join(__dirname,'dist/app.js'),'utf8');
for(const match of app.matchAll(/\$\('([\w-]+)'\)/g))assert(html.includes(`id="${match[1]}"`),`Missing DOM element ${match[1]}`);
assert(html.includes('CUNY SPS Capstone Project'));
assert(!html.includes('RESEARCH LAB'));assert(!app.includes('midnight only'));
assert(app.includes('z.id!==1'));assert(app.includes('origin+state.elapsed-3'));
console.log('PASS: all July origins and NYC zones, three correction modes, first-origin parity, delayed outcomes, imports, branding and DOM references.');
