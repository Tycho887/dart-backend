// Business Forms code shared by initial load, selection changes, and submission.
globalThis.dartEstimateForm ??= (() => {
  const state = {models: [], optimizers: [], busy: false, accepted: null};
  const value = (elements, id) => elements.find(e => e.id === id)?.value;
  const scalar = v => Array.isArray(v) ? v[0] : v;
  const ref = v => { const [name, version] = String(scalar(v)).split('@'); return {name, version: Number(version)}; };
  const absent = v => v === null || v === undefined || (typeof v === 'string' && v.trim() === '');
  function payload(elements) {
    const read = id => value(elements, id);
    const number = id => { const parsed = Number(read(id)); if (!Number.isFinite(parsed)) throw new Error(`Invalid number: ${id}`); return parsed; };
    const body = {
      contact_ids: String(read('contact_ids') ?? '').split(/[\s,]+/).filter(Boolean),
      ephemeris_id: String(read('ephemeris_id') ?? '').trim(),
      forward_model: ref(read('forward_model')), optimizer: ref(read('optimizer')),
      measurement_selection: {}, optimizer_overrides: {},
    };
    for (const id of ['min_elevation_deg', 'min_ebn0_db', 'min_abs_doppler_hz', 'max_abs_doppler_hz', 'min_samples_per_contact', 'doppler_sigma_hz']) {
      if (!absent(read(id))) body.measurement_selection[id] = number(id);
    }
    if (!absent(read('max_evaluations'))) body.optimizer_overrides.max_evaluations = number('max_evaluations');
    if (!absent(read('nominal_center_frequency_hz'))) body.nominal_center_frequency_hz = number('nominal_center_frequency_hz');
    if (!absent(read('label'))) body.label = String(read('label')).trim();
    return body;
  }
  async function request(path, body, key) {
    const response = await fetch('/dart/v1/' + path, {
      method: body === undefined ? 'GET' : 'POST', credentials: 'same-origin',
      headers: {'Content-Type': 'application/json', ...(key ? {'Idempotency-Key': key} : {})},
      body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.timeout(30000),
    });
    const text = await response.text();
    let result; try { result = JSON.parse(text); } catch { result = {}; }
    if (!response.ok) {
      const detail = result.errors?.map(e => `${e.location}: ${e.message}`).join('; ');
      throw new Error(detail || result.detail || text || `HTTP ${response.status}`);
    }
    return result;
  }
  function change(context) {
    const elements = context.panel.elements;
    const model = state.models.find(p => `${p.name}@${p.version}` === String(scalar(value(elements, 'forward_model'))));
    if (!model) { context.panel.disableSubmit(); return; }
    const compatible = state.optimizers.filter(p => p.compatible_models.includes(model?.name));
    const current = String(scalar(value(elements, 'optimizer')));
    const selected = compatible.find(p => `${p.name}@${p.version}` === current) || compatible[0];
    const updated = elements.map(e => {
      if (e.id === 'optimizer') return {...e, value: selected ? `${selected.name}@${selected.version}` : '', options: compatible.map(option)};
      if (e.id === 'summary') return {...e, value: model ? `${model.description} ${selected?.label ?? ''}. Locked samples only.` : 'Loading profiles…'};
      return e;
    });
    context.panel.onChangeElements(updated);
    let serialized;
    try { serialized = JSON.stringify(payload(updated)); }
    catch (error) {
      context.panel.onChangeElements(updated.map(e => e.id === 'summary' ? {...e, value: error.message} : e));
      context.panel.disableSubmit();
      return;
    }
    if (model && selected && !state.busy && serialized !== state.accepted) context.panel.enableSubmit();
    else context.panel.disableSubmit();
  }
  const option = p => ({id: `${p.name}@${p.version}`, label: `${p.label} (v${p.version})`, value: `${p.name}@${p.version}`, type: 'string'});
  async function load(context) {
    context.panel.disableSubmit();
    try {
      [state.models, state.optimizers] = await Promise.all([request('forward-model-profiles'), request('optimizer-profiles')]);
      const elements = context.panel.elements.map(e => e.id === 'forward_model' ? {...e, options: state.models.map(option)} : e);
      change({...context, panel: {...context.panel, elements}});
    } catch (error) { context.grafana.notifyError(['Profiles unavailable', error.message]); }
  }
  async function submit(context) {
    if (state.busy) return;
    state.busy = true; context.panel.disableSubmit();
    try {
      const body = payload(context.panel.elements);
      const serialized = JSON.stringify(body);
      if (serialized === state.accepted) return;
      const validated = await request('estimate-jobs/validate', body);
      if (!validated.valid) throw new Error('Request validation failed');
      const storageKey = 'dart.pendingEstimate';
      const pending = JSON.parse(sessionStorage.getItem(storageKey) || 'null');
      const key = pending?.body === serialized ? pending.key : crypto.randomUUID();
      sessionStorage.setItem(storageKey, JSON.stringify({body: serialized, key}));
      const result = await request('estimate-jobs', body, key);
      if (!result.estimate_uuid || !result.job_id) throw new Error('Submission response is incomplete; retry safely with the same request.');
      state.accepted = serialized; sessionStorage.removeItem(storageKey);
      context.grafana.locationService.partial({'var-estimate_uuid': result.estimate_uuid}, true);
      context.grafana.notifySuccess(['Estimate submitted', `${result.estimate_uuid} · ${result.status}`]);
      context.grafana.refresh();
    } catch (error) { context.grafana.notifyError(['Submission failed', error.message]); }
    finally { state.busy = false; change(context); }
  }
  const modelOptions = () => state.models.map(option);
  const optimizerOptions = elements => state.optimizers.filter(p => p.compatible_models.includes(ref(value(elements, 'forward_model')).name)).map(option);
  return {payload, load, change, submit, modelOptions, optimizerOptions};
})();
