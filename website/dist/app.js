(function(){
  'use strict';
  const C=window.ReplayCore,$=id=>document.getElementById(id);
  const state={data:null,origin:0,zone:186,elapsed:0,forecasts:new Map(),forecastSource:'full rolling correction',alpha:1,showBenchmark:false,zoneForecasts:new Map()};
  const NS='http://www.w3.org/2000/svg';
  let start,weather,eventMap;
  const int=n=>Number(n).toLocaleString('en-US');
  function element(tag,attrs,text){const e=document.createElementNS(NS,tag);for(const [k,v]of Object.entries(attrs||{}))e.setAttribute(k,String(v));if(text!==undefined)e.textContent=text;return e;}
  function dateAt(hour){return new Date(start+hour*C.HOUR);}
  function timeAt(hour){return dateAt(hour).toLocaleTimeString('en-US',{hour:'numeric',minute:'2-digit',timeZone:'UTC'});}
  function shortTime(hour){return dateAt(hour).toLocaleTimeString('en-US',{hour:'numeric',hour12:true,timeZone:'UTC'});}
  function dateLabel(hour){return dateAt(hour).toLocaleDateString('en-US',{month:'short',day:'numeric',timeZone:'UTC'});}
  function observedHour(){return originHour()+state.elapsed-3;}
  function hourRange(hour){return `${dateLabel(hour)} · ${shortTime(hour)}–${dateLabel(hour)===dateLabel(hour+1)?'':dateLabel(hour+1)+' '}${shortTime(hour+1)} EDT`;}
  function mapPopupText(id){const z=state.data.zones[id-1],hour=observedHour();return `${z.name} · ${z.borough}\n${hourRange(hour)}\n${int(C.value(state.data,id,hour))} observed pickups\nAvailable ${dateLabel(hour+3)} · ${shortTime(hour+3)} EDT (simulated)`;}
  function showMapPopup(id,x,y){const tip=$('map-tooltip'),box=$('zone-map').parentElement.getBoundingClientRect();tip.textContent=mapPopupText(id);tip.style.maxWidth=Math.min(280,box.width-24)+'px';tip.style.display='block';tip.style.left=Math.max(12,Math.min(x-box.left+12,box.width-tip.offsetWidth-12))+'px';tip.style.top=Math.max(12,Math.min(y-box.top+12,box.height-tip.offsetHeight-12))+'px';}
  function isoAt(hour){return dateAt(hour).toISOString().slice(0,16);}
  function notify(message){$('notification').textContent=message;}
  function zone(){return state.data.zones[state.zone-1];}
  function originHour(){return C.originHour(state.origin,state.data);}
  function rows(){
    const data=C.sixHours(state.data,state.zone,state.origin,state.elapsed,state.forecasts);
    const z=state.zoneForecasts.get(state.zone);
    return data.map(row=>({...row,prediction:state.forecastSource==='CSV import'?row.prediction:C.adjustedForecast(z,state.origin,row.horizon,state.alpha)}));
  }
  function originChanged(index){state.origin=Math.max(0,Math.min(737,index));state.elapsed=0;notify('');render();}
  function setZone(id){state.zone=Number(id);$('zone-search').value='';populateZones();render();}
  function populateZones(){
    const query=$('zone-search').value.trim().toLowerCase(),select=$('zone-select');select.replaceChildren();
    const filtered=state.data.zones.filter(z=>z.id!==1).filter(z=>`${z.name} ${z.borough} ${z.id}`.toLowerCase().includes(query));
    for(const z of filtered){const option=document.createElement('option');option.value=z.id;option.textContent=z.name+' · '+z.borough;select.append(option);}
    if(!filtered.length){const option=document.createElement('option');option.textContent='No matching zones';option.disabled=true;select.append(option);}
    if(filtered.some(z=>z.id===state.zone))select.value=state.zone;
    else select.selectedIndex=-1;
  }
  function buildMap(){
    const svg=$('zone-map');
    for(const [id,d]of Object.entries(state.data.mapPaths)){
      const path=element('path',{d,'data-zone':id,tabindex:'0',role:'button','aria-label':state.data.zones[Number(id)-1].name});
      path.addEventListener('click',e=>{setZone(id);showMapPopup(Number(id),e.clientX,e.clientY);});
      path.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();setZone(id);}});
      path.addEventListener('pointermove',e=>showMapPopup(Number(id),e.clientX,e.clientY));
      path.addEventListener('focus',()=>{const box=path.getBoundingClientRect();showMapPopup(Number(id),box.x+box.width/2,box.y+box.height/2);});
      path.addEventListener('blur',()=>{$('map-tooltip').style.display='none';});
      path.addEventListener('pointerleave',()=>{$('map-tooltip').style.display='none';});svg.append(path);
    }
    const labels=[['MANHATTAN',318,279,-59],['BRONX',494,145,0],['QUEENS',536,377,0],['BROOKLYN',361,496,0],['STATEN ISLAND',132,581,0]];
    for(const [text,x,y,rotation]of labels)svg.append(element('text',{x,y,class:'map-label',transform:`rotate(${rotation} ${x} ${y})`,'text-anchor':'middle'},text));
  }
  function renderMap(){
    const previous=observedHour();
    $('map-tooltip').style.display='none';
    $('map-observed-period').textContent=`Observed pickups · ${hourRange(previous)}`;
    const values=state.data.counts.map(v=>v[previous]);const max=Math.max(...values);
    for(const path of $('zone-map').querySelectorAll('path')){
      const id=Number(path.dataset.zone),ratio=Math.sqrt(values[id-1]/Math.max(1,max));
      const lo=[231,237,219],hi=[66,105,73];path.style.fill=`rgb(${lo.map((v,i)=>Math.round(v+(hi[i]-v)*ratio)).join(',')})`;
      path.classList.toggle('selected',id===state.zone);path.setAttribute('aria-pressed',String(id===state.zone));path.setAttribute('aria-description',mapPopupText(id));
    }
    $('selected-zone-name').textContent=zone().name;
    $('forecast-zone-context').textContent=`${zone().name} · ${zone().borough}`;
    $('selected-zone-meta').textContent=`${zone().borough} · TLC zone ${state.zone}`+(state.zone===1?' · Outside NYC map':'');
  }
  function renderChart(data){
    const svg=$('outlook-chart');svg.replaceChildren();
    const visibleValues=data.flatMap(r=>state.showBenchmark?[r.reference,r.actual,r.prediction]:[r.actual,r.prediction]).filter(v=>v!==null);
    const highest=Math.max(10,...visibleValues),rough=highest*1.25/4;
    const magnitude=10**Math.floor(Math.log10(rough)),fraction=rough/magnitude;
    const step=(fraction<=1?1:fraction<=2?2:fraction<=5?5:10)*magnitude,top=step*4;
    const left=49,right=650,upper=20,bottom=225,x=i=>left+i*(right-left)/5,y=v=>bottom-v/top*(bottom-upper);
    for(let i=0;i<=4;i++){const value=i*step;svg.append(element('line',{x1:left,y1:y(value),x2:right,y2:y(value),class:'chart-grid'}));svg.append(element('text',{x:left-12,y:y(value)+4,'text-anchor':'end',class:'chart-tick'},int(value)));}
    for(let i=0;i<6;i++)svg.append(element('text',{x:x(i),y:253,'text-anchor':'middle',class:'chart-tick'},shortTime(data[i].target)));
    // Shade hours whose outcomes remain hidden; future actuals never enter SVG attributes.
    const firstHidden=data.findIndex(r=>r.actual===null);
    if(firstHidden>=0){const shadeX=firstHidden===0?left:x(firstHidden)-25;svg.append(element('rect',{x:shadeX,y:upper,width:right-shadeX,height:bottom-upper,fill:'#f1f4e9',opacity:'.35'}));}
    function series(key,css,color){
      let segment=[];
      function flush(){if(segment.length>1)svg.append(element('polyline',{points:segment.map(([i,v])=>`${x(i)},${y(v)}`).join(' '),class:css}));segment=[];}
      data.forEach((row,i)=>{if(row[key]===null){flush();return;}segment.push([i,row[key]]);});flush();
      data.forEach((row,i)=>{if(row[key]===null)return;const dot=element('circle',{cx:x(i),cy:y(row[key]),r:key==='actual'?4.3:3.3,fill:color,stroke:'#fff','stroke-width':'1.5'});dot.append(element('title',{},`${shortTime(row.target)} · ${key==='reference'?'Same hour last week':key==='actual'?'Actual pickups':'Selected forecast'}: ${row[key]}`));svg.append(dot);});
    }
    if(state.showBenchmark)series('reference','chart-baseline','#92a08b');series('prediction','chart-model','#35659a');series('actual','chart-actual','#d78d4c');
    svg.setAttribute('aria-label',`Six-hour view for ${zone().name}. ${state.showBenchmark?'Weekly benchmark: '+data.map(r=>r.reference).join(', ')+'.':''} XGBoost forecasts: ${data.map(r=>r.prediction===null?'not generated':r.prediction.toFixed(1)).join(', ')}. ${data.filter(r=>r.actual!==null).length} actual hours revealed.`);
  }
  function renderHistory(){
    const svg=$('history-chart'),last=observedHour();
    svg.replaceChildren();
    const values=Array.from({length:24},(_,i)=>C.value(state.data,state.zone,last-23+i));
    const top=Math.max(10,...values)*1.15;
    const x=i=>48+i*606/23,y=v=>175-v/top*145;
    for(let i=0;i<4;i++){
      const value=top*i/3;
      svg.append(element('line',{x1:48,x2:654,y1:y(value),y2:y(value),class:'chart-grid'}));
      svg.append(element('text',{x:40,y:y(value)+4,'text-anchor':'end',class:'chart-tick'},int(Math.round(value))));
    }
    svg.append(element('polyline',{points:values.map((v,i)=>`${x(i)},${y(v)}`).join(' '),fill:'none',stroke:'#285e4c','stroke-width':2.5}));
    for(const i of [0,6,12,18,23])svg.append(element('text',{x:x(i),y:205,'text-anchor':'middle',class:'chart-tick'},shortTime(last-23+i)));
    $('history-note').textContent=`${zone().name} · Latest count: ${int(values[23])} pickups for ${hourRange(last)}. Each count is released two hours after its hour ends.`;
    svg.setAttribute('aria-label',`Available history for ${zone().name}: ${values.join(', ')} pickups. Last hour ${hourRange(last)}.`);
  }
  function renderTable(data){
    const body=$('hour-table');body.replaceChildren();
    for(const row of data){
      const tr=document.createElement('tr'),hour=document.createElement('td');hour.className='hour-label';hour.textContent=shortTime(row.target);
      const label=document.createElement('small');label.textContent=`+${row.horizon}h`;hour.append(label);tr.append(hour);
      const prediction=document.createElement('td');prediction.textContent=row.prediction===null?'—':row.prediction.toFixed(1);if(row.prediction===null)prediction.className='pending';tr.append(prediction);
      const reference=document.createElement('td');reference.textContent=int(row.reference);reference.className='benchmark-column';reference.hidden=!state.showBenchmark;tr.append(reference);
      const actual=document.createElement('td');actual.textContent=row.actual===null?'Not yet revealed':int(row.actual);if(row.actual===null)actual.className='unrevealed';tr.append(actual);body.append(tr);
    }
  }
  function render(){
    const origin=originHour(),data=rows(),time=isoAt(origin);
    $('origin-date').value=time.slice(0,10);$('origin-hour').value=dateAt(origin).getUTCHours();
    for(const option of $('origin-hour').options)option.disabled=time.slice(0,10)==='2026-07-31'&&Number(option.value)>17;
    $('previous-origin').disabled=state.origin===0;$('next-origin').disabled=state.origin===737;
    $('zone-select').value=state.zone;
    const first=data[0],available=first.actual!==null;
    $('last-count').textContent=available?int(first.actual):'Pending';
    $('last-period').textContent=hourRange(first.target);
    $('last-availability').textContent=available
      ?`Released at ${dateLabel(first.target+3)} · ${timeAt(first.target+3)} EDT. Same hour as the forecast.`
      :`Available at ${dateLabel(first.target+3)} · ${timeAt(first.target+3)} EDT. Advance the clock to reveal this count.`;
    const error=available&&first.prediction!==null?first.actual-first.prediction:null;
    $('forecast-window').textContent=error===null?'Pending':`${error>0?'+':''}${error.toFixed(1)}`;
    $('forecast-window-note').textContent=error===null
      ?'The difference appears when this hour’s actual count is released.'
      :error===0?'Forecast matched the actual count.':error>0?'More pickups than forecast.':'Fewer pickups than forecast.';
    const predicted=data.filter(r=>r.prediction!==null);
    $('model-summary').textContent=data[0].prediction!==null?data[0].prediction.toFixed(1):'Not generated';
    $('model-unit').textContent=data[0].prediction!==null?'pickups':'';
    $('model-note').textContent=data[0].prediction!==null?`${hourRange(first.target)} · Made ${dateLabel(origin)} at ${timeAt(origin)} EDT · ${state.forecastSource}`:'This origin has no model forecast yet.';
    $('forecast-legend-note').textContent=predicted.length?`${predicted.length}/6 hours`:'not generated';
    $('forecast-banner-title').textContent=predicted.length?'XGBoost forecasts are ready for this origin.':'No XGBoost forecasts for this origin yet.';
    $('forecast-banner-note').textContent=state.forecastSource==='CSV import'?'The displayed model values came from your forecast CSV. Actuals follow the simulated two-hour publication delay.':'All 738 prepared July origins have six forecasts for each of 262 NYC zones. The last origin is July 31 at 5 PM so target hours remain within July. This is a retrospective simulation, not a live feed.';
    const zoneForecast=state.zoneForecasts.get(state.zone);
    $('correction-note').textContent=state.forecastSource==='CSV import'?'Imported predictions; no additional rolling adjustment.':`${state.alpha===0?'Original model':state.alpha===1?'Full':'25% damped'} · zone adjustment ${(state.alpha*zoneForecast.correction[state.origin]).toFixed(1)} pickups · ${zoneForecast.historyCount[state.origin]} available hourly residuals`;
    $('show-model-origin').disabled=state.origin===0;
    $('clock-display').textContent=`${dateLabel(origin+state.elapsed)} · ${timeAt(origin+state.elapsed)}`;
    $('clock-range').value=state.elapsed;$('clock-step').disabled=state.elapsed===9;$('clock-reset').disabled=state.elapsed===0;
    const revealed=data.filter(r=>r.actual!==null).length;
    $('reveal-message').textContent=revealed?`${revealed} of 6 target outcomes available. Actuals appear two hours after each hour ends.`:'Actuals appear two hours after each target hour ends.';
    $('lag-day').textContent=int(C.value(state.data,state.zone,origin+1-24));$('lag-week').textContent=int(data[0].reference);$('lag-two-week').textContent=int(C.value(state.data,state.zone,origin+1-336));
    const w=weather.get(zone().weatherClass),position=state.origin*6;
    $('weather-temperature').textContent=w.t[position].toFixed(1)+'°F';
    const eventMask=eventMap.get(`${origin+1}:${state.zone}`)||0;
    $('weather-precipitation').textContent=Number.isFinite(w.p[position])?w.p[position].toFixed(2)+' in':'Unavailable';
    $('sports-context').textContent=eventMask&1?'Post-game window':'No flagged window';
    $('concert-context').textContent=eventMask&6?'Estimated concert window':'No flagged window';
    let context=`For ${hourRange(origin+1)} · forecast run ${state.data.runAges[state.origin]}h old · ${w.w[position].toFixed(1)} mph wind`;
    $('input-context').textContent=context;
    for(const node of document.querySelectorAll('.benchmark-column'))node.hidden=!state.showBenchmark;
    renderMap();renderChart(data);renderTable(data);renderHistory();
  }
  function exportView(){
    const header=['forecast_origin_utc','location_id','forecast_horizon','target_hour_local','same_hour_last_week','prediction','original_prediction','rolling_correction','correction_strength','revealed_actual','simulation_clock_local'];
    const originUtc=new Date(start+(originHour()+4)*C.HOUR).toISOString();
    const lines=rows().map(r=>[originUtc,state.zone,r.horizon,isoAt(r.target),r.reference,r.prediction??'',state.zoneForecasts.get(state.zone).original[state.origin*6+r.horizon-1],state.forecastSource==='CSV import'?'':state.zoneForecasts.get(state.zone).correction[state.origin],state.forecastSource==='CSV import'?'':state.alpha,r.actual??'',isoAt(originHour()+state.elapsed)].join(','));
    const blob=new Blob([header.join(',')+'\n'+lines.join('\n')+'\n'],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`july-zone-${state.zone}-origin-${isoAt(originHour()).replace(/[:T]/g,'-')}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    notify('Downloaded the six-hour view. Hidden actuals and missing forecasts are blank.');
  }
  const panels={
    dataset:()=>`<h2>CUNY SPS Capstone Project</h2><p>Farhod Ibragimov · MS in Data Science. A historical July 2026 replay of NYC completed-pickup forecasts.</p><div class="dialog-stats"><div><strong>1,160,136</strong>forecast cases</div><div><strong>738</strong>hourly origins</div><div><strong>30</strong>model inputs</div><div><strong>262</strong>NYC zones</div><div><strong>6</strong>hourly horizons</div></div><h3>July coverage</h3><p>Origins run from July 1 at midnight to July 31 at 5 PM, New York time. The final six issuance hours are excluded because their complete six-hour targets would extend beyond July. EWR is excluded from this replay.</p><h3>Model provenance</h3><p>The frozen model was trained with 263 zones; this display and correction use 262 NYC zones. The model artifact was created retrospectively, so these are simulated historical forecasts, not predictions actually issued in July.</p>`,
    methods:()=>`<h2>Forecast, correct, then reveal.</h2><p>Choose any prepared July origin and zone. Original XGBoost, 25% damped correction and full rolling correction are available for every origin. The full correction is the default candidate because reducing underprediction is a project priority; it is not a validated operational policy.</p><h3>How corrections work</h3><p>For each zone, the correction averages the preceding seven days of available horizon-1 errors: actual minus original XGBoost prediction. The same adjustment is applied to all six horizons, with a zero lower bound. Late-June forecasts initialize July; initial coverage is 165 hours rather than 168. Gaps remain explicit through the displayed history count. No model retraining occurs.</p><h3>Simulated pickup availability</h3><p>A pickup hour labeled 10 AM covers 10–11 AM. Its count becomes available at 1 PM, two hours after it ends. Corrections use only outcomes available by forecast issuance and remain fixed as you advance that issuance's clock. Actuals and the map follow this same delay; advance nine hours to reveal all six outcomes.</p><h3>Inputs and limitations</h3><p>The model uses the original 30 ordered features, including earlier pickup counts, archived forecast weather, calendar and estimated event windows. Weather retains the seven-hour eligibility assumption. Retrospective event as-of availability is not established. Completed pickups count rides that went ahead. Someone who leaves the app after seeing a high fare or a long wait is outside that count. No real-time pickup feed or driver-supply data is connected.</p><h3>Historical replay</h3><p>Aggregate outcomes are bundled with this static website. Hiding future actuals is presentation behavior, not a data-security boundary. July is an exploratory replay, not a fresh untouched final test. Forecasts are precomputed in Python; the browser selects the correction mode. CSV imports are used as supplied without another correction.</p>`,
    sources:()=>`<h2>The sources behind the replay.</h2><p>This site contains aggregate zone-hour counts and prepared feature summaries. No raw trip records or credentials are bundled.</p><ul><li><a href="https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page" target="_blank" rel="noopener">NYC TLC high-volume FHV trip records and zone reference</a> — completed pickups.</li><li><a href="https://open-meteo.com/en/docs/single-runs-api" target="_blank" rel="noopener">Open-Meteo Single Runs API</a> — archived forecast weather.</li><li><a href="https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/teams/ny/schedule" target="_blank" rel="noopener">ESPN schedule endpoints</a> — historical sports schedules.</li><li><a href="https://api.setlist.fm/docs/1.0/index.html" target="_blank" rel="noopener">setlist.fm API</a> — concert dates and venues.</li></ul><p>The trip records count rides that went ahead. The website does not estimate individual passengers switching from the subway. The causal analysis remains a separate part of the capstone.</p>`
  };
  function openPanel(name){$('dialog-content').innerHTML=panels[name]();$('info-dialog').showModal();}
  function setup(){
    for(let hour=0;hour<24;hour++){const option=document.createElement('option');option.value=hour;option.textContent=new Date(Date.UTC(2026,6,1,hour)).toLocaleTimeString('en-US',{hour:'numeric',minute:'2-digit',timeZone:'UTC'});$('origin-hour').append(option);}
    $('show-benchmark').addEventListener('change',e=>{state.showBenchmark=e.target.checked;render();});
    $('forecast-mode').addEventListener('change',e=>{state.alpha=Number(e.target.value);state.forecastSource=state.alpha===0?'original XGBoost':state.alpha===1?'full rolling correction':'25% damped correction';render();});
    populateZones();buildMap();render();
    $('origin-date').addEventListener('change',()=>{const value=$('origin-date').value;if(!/^2026-07-\d{2}$/.test(value)){render();return;}originChanged(C.originIndex(value+'T'+String($('origin-hour').value).padStart(2,'0')+':00',state.data));});
    $('origin-hour').addEventListener('change',()=>originChanged(C.originIndex($('origin-date').value+'T'+String($('origin-hour').value).padStart(2,'0')+':00',state.data)));
    $('previous-origin').addEventListener('click',()=>originChanged(state.origin-1));$('next-origin').addEventListener('click',()=>originChanged(state.origin+1));
    $('zone-search').addEventListener('input',populateZones);$('zone-select').addEventListener('change',e=>setZone(e.target.value));
    $('clock-range').addEventListener('input',e=>{state.elapsed=Number(e.target.value);render();});
    $('clock-step').addEventListener('click',()=>{state.elapsed=Math.min(9,state.elapsed+1);render();});$('clock-reset').addEventListener('click',()=>{state.elapsed=0;render();});
    $('export-view').addEventListener('click',exportView);
    $('show-model-origin').addEventListener('click',()=>originChanged(0));
    for(const button of document.querySelectorAll('[data-panel]'))button.addEventListener('click',()=>openPanel(button.dataset.panel));
    $('replay-nav').addEventListener('click',()=>window.scrollTo({top:0,behavior:'smooth'}));$('close-dialog').addEventListener('click',()=>$('info-dialog').close());
    $('info-dialog').addEventListener('click',e=>{if(e.target===$('info-dialog')){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();}});
    $('forecast-file').addEventListener('change',async e=>{const file=e.target.files[0];if(!file)return;try{if(file.size>80*1024*1024)throw Error('Please use a CSV smaller than 80 MB.');const imported=C.importForecasts(await file.text(),state.data);state.forecasts=imported;state.forecastSource='CSV import';render();notify(`Loaded ${int(imported.size)} user-provided forecast cases. No additional model inference was run by this website.`);}catch(error){notify('Forecast import failed: '+error.message);}finally{e.target.value='';}});
    const query=new URLSearchParams(location.search),qzone=Number(query.get('zone'));if(qzone>=2&&qzone<=263)setZone(qzone);
    $('loading').classList.add('hidden');setTimeout(()=>$('loading').remove(),350);
  }
  Promise.all(['replay-data.json','july-forecasts.json'].map(file=>fetch(file).then(response=>{if(!response.ok)throw Error(file+' returned '+response.status);return response.json();}))).then(([data,forecasts])=>{
    if(forecasts.originCount!==738||forecasts.cases!==1160136||forecasts.zones.length!==262)throw Error('Unexpected July forecast coverage.');
    for(const z of forecasts.zones){if(z.id<2||z.id>263||state.zoneForecasts.has(z.id)||z.original.length!==738*6||z.correction.length!==738||z.historyCount.length!==738||!z.original.every(v=>Number.isFinite(v)&&v>=0)||!z.correction.every(Number.isFinite))throw Error('Invalid July zone forecasts.');state.zoneForecasts.set(z.id,z);}
    state.data=data;start=C.floatingTime(data.startLocal);weather=new Map(data.weather.map(w=>[w.class,w]));eventMap=new Map(data.events.map(([hour,zone,mask])=>[`${hour}:${zone}`,mask]));setup();
  }).catch(error=>{$('loading').replaceChildren();const title=document.createElement('h2');title.textContent='The replay could not load.';const text=document.createElement('p');text.textContent='Please reload this page. '+error.message;$('loading').append(title,text);});
})();
