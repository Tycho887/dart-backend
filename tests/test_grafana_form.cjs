// Exercise the actual shipped Business Forms code without contacting any services.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('deploy/grafana/form.js', 'utf8');
function fixture(fetch) {
  const storage = new Map();
  const sandbox = {fetch, AbortSignal, crypto: require('node:crypto').webcrypto,
    sessionStorage: {getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v),removeItem:k=>storage.delete(k)}};
  vm.runInNewContext(source,sandbox);
  const messages = [];
  const panel = {elements: Object.entries({contact_ids:'a, b',ephemeris_id:'prior',forward_model:'lofi-time@1',optimizer:'least-squares@1',min_elevation_deg:0,min_ebn0_db:0}).map(([id,value])=>({id,value})),
    enableSubmit(){},disableSubmit(){},onChangeElements(elements){panel.elements=elements;}};
  const context = {panel,grafana:{notifyError:x=>messages.push(x),notifySuccess(){},refresh(){},locationService:{partial(){}}}};
  return {form:sandbox.dartEstimateForm,context,storage,messages};
}
const response = body => ({ok:true,text:async()=>JSON.stringify(body)});
test('payload retains zeros and optional omission',()=>{
  const {form,context}=fixture(); const body=form.payload(context.panel.elements);
  assert.equal(body.measurement_selection.min_elevation_deg,0);
  assert.equal(body.measurement_selection.min_ebn0_db,0);
  assert.equal(body.nominal_center_frequency_hz,undefined);
  assert.deepEqual(Array.from(body.contact_ids),['a','b']);
  context.panel.elements.push({id:'nominal_center_frequency_hz',value:'   '});
  assert.equal(form.payload(context.panel.elements).nominal_center_frequency_hz,undefined);
  context.panel.elements.at(-1).value='invalid';
  assert.throws(()=>form.payload(context.panel.elements),/Invalid number/);
});
test('uncertain submission retries retain the idempotency key',async()=>{
  let fail=true;const keys=[];
  const {form,context,storage}=fixture(async(url,options)=>{
    if(url.endsWith('/validate'))return response({valid:true});
    keys.push(options.headers['Idempotency-Key']);
    if(fail){fail=false;throw new Error('connection lost');}
    return response({estimate_uuid:'estimate',job_id:'job',status:'queued'});
  });
  await form.submit(context);assert.equal(storage.size,1);
  await form.submit(context);assert.equal(keys[0],keys[1]);assert.equal(storage.size,0);
  await form.submit(context);assert.equal(keys.length,2);
});
test('profile load filters optimizer choices',async()=>{
 const {form,context}=fixture(async url=>response(url.includes('forward-model')?
   [{name:'lofi-time',version:1,label:'Time',description:'Time fit'}]:
   [{name:'least-squares',version:1,label:'LS',compatible_models:['lofi-time']},{name:'phase',version:1,label:'Phase',compatible_models:['lofi-elements']}]
 ));
 await form.load(context);
 assert.equal(context.panel.elements.find(e=>e.id==='forward_model').options.length,1);
 assert.equal(context.panel.elements.find(e=>e.id==='optimizer').options.length,1);
 assert.equal(form.modelOptions()[0].value,'lofi-time@1');
 assert.equal(form.optimizerOptions(context.panel.elements)[0].value,'least-squares@1');
 context.panel.elements.push({id:'summary',value:''},{id:'max_evaluations',value:'invalid'});
 assert.doesNotThrow(()=>form.change(context));
 assert.match(context.panel.elements.find(e=>e.id==='summary').value,/Invalid number/);
});
