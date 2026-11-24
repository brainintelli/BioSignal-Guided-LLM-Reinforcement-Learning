
"use strict";
const report = JSON.parse(document.getElementById('report-data').textContent);
const ranked = report.arms.slice().sort((a,b)=>a.rank-b.rank);
document.querySelectorAll('.filter').forEach(button=>button.addEventListener('click',()=>{
  const filter=button.dataset.filter;
  document.querySelectorAll('.filter').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});
  let count=0;document.querySelectorAll('#matrix-table tbody tr').forEach(row=>{row.hidden=filter!=='全部' && row.dataset.category!==filter;if(!row.hidden) count++;});
  document.getElementById('matrix-count').textContent=`显示${count}组`;
}));
const NS='http://www.w3.org/2000/svg';
function svgel(tag,attrs,text){const el=document.createElementNS(NS,tag);for(const [key,value] of Object.entries(attrs||{}))el.setAttribute(key,value);if(text!==undefined)el.textContent=text;return el;}
function renderChart(){
  const metric=document.getElementById('chart-metric').value;
  const key=metric==='rmst'?'rmst_days':metric==='success'?'successes':'successes_10day';
  const max=metric==='rmst'?30:16, suffix=metric==='rmst'?'天':'局',W=920,H=605,left=130,right=90,plot=W-left-right;
  const root=svgel('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-labelledby':'chart-title chart-desc'});
  root.append(svgel('title',{id:'chart-title'},metric==='rmst'?'10组方案训练前后平均存活时间':`${metric==='success'?'30':'10'}天存活局数对比`));
  root.append(svgel('desc',{id:'chart-desc'},'按冻结筛选排名排列，灰色为训练前，绿色为训练8轮后。完整数值可在后续数据表中读取。'));
  for(let n=0;n<=4;n++){const value=max*n/4,x=left+plot*n/4;root.append(svgel('line',{x1:x,y1:36,x2:x,y2:H-30,stroke:'#dce2d7','stroke-dasharray':n?'3 5':'0'}));root.append(svgel('text',{x,y:23,'text-anchor':'middle',fill:'#657269','font-size':12},String(value)));}
  ranked.forEach((arm,index)=>{const y=55+index*54;
    root.append(svgel('text',{x:left-15,y:y+15,'text-anchor':'end','font-size':14,fill:'#203a32'},`${arm.rank}. ${arm.short}`));
    [['initial','#c7d1c6',0],['after','#1b6a57',19]].forEach(([phase,color,offset])=>{
      const value=arm[phase][key],width=plot*value/max;
      const rect=svgel('rect',{x:left,y:y+offset,width,height:13,rx:2,fill:color});rect.append(svgel('title',{},`${arm.title} ${phase==='initial'?'训练前':'训练后'}：${metric==='rmst'?value.toFixed(2):value}${suffix}`));root.append(rect);
      root.append(svgel('text',{x:left+width+7,y:y+offset+11,'font-size':12,fill:phase==='after'?'#1b6a57':'#657269'},metric==='rmst'?value.toFixed(2):`${value}/16`));
    });
  });
  document.getElementById('results-chart').replaceChildren(root);
}
document.getElementById('chart-metric').addEventListener('change',renderChart);renderChart();
document.querySelectorAll('.map-mode').forEach(button=>button.addEventListener('click',()=>{
  const freeze=button.dataset.mode==='freeze';document.querySelectorAll('.map-mode').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});
  document.getElementById('resource-clock').textContent=freeze?'90刻':'0刻／已恢复';
  document.getElementById('map-explanation').textContent=freeze?'冻结模式：R区局部时钟不动，返回后还要在活跃范围内推进90刻。玩家并未随资源一起暂停。':'全局模式：世界已经过去100刻，超过剩余90刻，返回时资源已恢复。角色的饥渴与生命照常结算。';
}));
let printDetails=[];
window.addEventListener('beforeprint',()=>{printDetails=[...document.querySelectorAll('details')].map(el=>[el,el.open]);printDetails.forEach(([el])=>el.open=true);});
window.addEventListener('afterprint',()=>{printDetails.forEach(([el,open])=>el.open=open);});
document.querySelectorAll('.print-button').forEach(button=>button.addEventListener('click',()=>window.print()));
document.documentElement.dataset.reportReady='true';
