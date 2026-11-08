(function(root){
  'use strict';
  const HOUR=3600000;
  const core={
    HOUR,
    floatingTime(value){return Date.parse(value.endsWith('Z')?value:value+'Z');},
    originIndex(value,data){return Math.round((this.floatingTime(value)-this.floatingTime(data.firstOriginLocal))/HOUR);},
    originHour(index,data){return data.firstOriginHour+index;},
    isRevealed(horizon,elapsed){return elapsed>=horizon+3;},
    value(data,zone,hour){if(zone<1||zone>263||hour<0||hour>=data.hours)return null;return data.counts[zone-1][hour];},
    sixHours(data,zone,index,elapsed,forecasts){
      const origin=this.originHour(index,data);
      return Array.from({length:6},(_,i)=>{
        const horizon=i+1,target=origin+horizon;
        const key=[index,zone,horizon].join(':');
        return {horizon,target,reference:this.value(data,zone,target-168),
          actual:this.isRevealed(horizon,elapsed)?this.value(data,zone,target):null,
          prediction:forecasts&&forecasts.has(key)?forecasts.get(key):null};
      });
    },
    adjustedForecast(zoneData,index,horizon,alpha){
      if(!zoneData)return null;
      const original=zoneData.original[index*6+horizon-1];
      const correction=zoneData.correction[index];
      return Math.max(0,original+alpha*correction);
    },
    parseCsv(text){
      const rows=[];let row=[],value='',quoted=false;
      text=text.replace(/^\uFEFF/,'');
      for(let i=0;i<text.length;i++){
        const c=text[i];
        if(c==='"'){if(quoted&&text[i+1]==='"'){value+='"';i++;}else quoted=!quoted;}
        else if(c===','&&!quoted){row.push(value);value='';}
        else if((c==='\n'||c==='\r')&&!quoted){if(c==='\r'&&text[i+1]==='\n')i++;row.push(value);if(row.some(v=>v.trim()))rows.push(row);row=[];value='';}
        else value+=c;
      }
      if(quoted)throw Error('The CSV has an unclosed quoted field.');
      row.push(value);if(row.some(v=>v.trim()))rows.push(row);
      return rows;
    },
    importForecasts(text,data){
      const rows=this.parseCsv(text);if(rows.length<2)throw Error('The file needs a header and at least one forecast row.');
      const header=rows[0].map(v=>v.trim());
      const required=['forecast_origin_utc','location_id','forecast_horizon','prediction'];
      for(const name of required)if(!header.includes(name))throw Error('Missing CSV column: '+name);
      const indexes=Object.fromEntries(required.map(name=>[name,header.indexOf(name)]));
      const forecasts=new Map();
      for(let i=1;i<rows.length;i++){
        const row=rows[i];
        if(row.length!==header.length)throw Error('CSV row '+(i+1)+' has the wrong number of fields.');
        const rawOrigin=row[indexes.forecast_origin_utc].trim();
        if(!/(Z|[+-]\d{2}:?\d{2})$/i.test(rawOrigin))throw Error('CSV origins need an explicit UTC offset or Z.');
        const utc=Date.parse(rawOrigin.replace(' ','T'));
        const firstUtc=Date.parse(data.firstOriginLocal+'-04:00');
        const index=(utc-firstUtc)/HOUR;
        const zone=Number(row[indexes.location_id]),h=Number(row[indexes.forecast_horizon]);
        const raw=row[indexes.prediction].trim(),prediction=Number(raw);
        if(!Number.isInteger(index)||index<0||index>=data.originCount)throw Error('Row '+(i+1)+': origin is outside this replay.');
        if(!Number.isInteger(zone)||zone<2||zone>263||!Number.isInteger(h)||h<1||h>6)throw Error('Row '+(i+1)+': invalid zone or horizon.');
        if(!raw||!Number.isFinite(prediction)||prediction<0)throw Error('Row '+(i+1)+': prediction must be finite and nonnegative.');
        const key=[index,zone,h].join(':');if(forecasts.has(key))throw Error('Duplicate forecast case at row '+(i+1)+'.');
        forecasts.set(key,prediction);
      }
      return forecasts;
    }
  };
  if(typeof module==='object'&&module.exports)module.exports=core;
  else root.ReplayCore=core;
})(typeof window!=='undefined'?window:globalThis);
