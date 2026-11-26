'use strict';

const byId = (id) => document.getElementById(id);
const rubricIds = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7'];
const shortNames = ['EIC', 'GMP', 'CJC', 'Penetr.', 'CIC', 'SMD', 'IF'];

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function conditionImage(path, alt, lazy = false) {
  const image = element('img');
  image.src = path;
  image.alt = alt;
  image.decoding = 'async';
  if (lazy) image.loading = 'lazy';
  return image;
}

function videoPlayer(example, label) {
  const video = element('video');
  video.controls = true;
  video.muted = true;
  video.playsInline = true;
  video.preload = 'none';
  video.poster = example.poster;
  video.src = example.video;
  video.setAttribute('aria-label', label);
  video.append('Your browser does not support this video.');
  return video;
}

function scoreList(scores) {
  const row = element('div', 'score-list');
  rubricIds.forEach((id) => {
    const value = scores[id];
    const score = element('span', 'score-cell', `${id} ${Number.isInteger(value) ? value : value.toFixed(2)}`);
    if (value === 0) score.classList.add('fail');
    else if (value < 1) score.classList.add('mixed');
    row.append(score);
  });
  return row;
}

function showComparison(example) {
  const container = element('div', 'compare-inner');
  const condition = element('div', 'condition');
  const text = element('div');
  text.append(element('div', 'media-label', `Initial frame · ${example.task_id}`), element('p', '', example.caption));
  condition.append(conditionImage(example.image, 'Initial frame shared by all shown rollouts'), text);

  const controls = element('div', 'compare-tools');
  const play = element('button', 'button small', 'Play together');
  play.type = 'button';
  const comparisonPass = example.pass_id ?? example.videos[0].pass_id;
  controls.append(play, element('span', '', comparisonPass
    ? `WorldPhyReward scores · pass ${comparisonPass}`
    : 'WorldPhyReward scores'));
  const grid = element('div', 'video-grid');
  const players = [];
  example.videos.forEach((clip) => {
    const card = element('article', 'video-card');
    const video = videoPlayer(clip, `${clip.model} generated rollout`);
    players.push(video);
    const meta = element('div', 'video-meta');
    const total = clip.total_score ?? Object.values(clip.scores).reduce((sum, value) => sum + value, 0);
    meta.append(element('strong', '', clip.model), scoreList(clip.scores),
      element('p', 'video-total', `${Number(total.toFixed(2))} / 7 criteria`));
    card.append(video, meta);
    grid.append(card);
  });
  const updatePlay = () => {
    play.textContent = players.every((video) => video.paused) ? 'Play together' : 'Pause all';
  };
  players.forEach((video) => {
    video.addEventListener('play', updatePlay);
    video.addEventListener('pause', updatePlay);
    video.addEventListener('ended', updatePlay);
  });
  play.addEventListener('click', () => {
    if (players.some((video) => !video.paused)) {
      players.forEach((video) => video.pause());
    } else {
      players.forEach((video) => {
        video.currentTime = 0;
        video.play().catch(() => updatePlay());
      });
    }
  });
  container.append(condition, controls, grid);
  byId('comparison').querySelectorAll('video').forEach((video) => video.pause());
  byId('comparison').replaceChildren(container);
}

function setupComparisons(examples) {
  const select = byId('compare-select');
  examples.forEach((example, index) => {
    const option = element('option', '', example.title ? `${example.title} · ${example.task_id}` : example.task_id);
    option.value = index;
    select.append(option);
  });
  byId('compare-picker').hidden = examples.length < 2;
  select.addEventListener('change', () => showComparison(examples[Number(select.value)]));
  showComparison(examples[0]);
}

function setupRubrics(examples) {
  const parent = byId('rubric-tabs');
  const panel = byId('rubric-panel');
  const tabs = examples.map((example, index) => {
    const label = shortNames[rubricIds.indexOf(example.id)];
    const tab = element('button');
    tab.type = 'button';
    tab.id = `tab-${example.id}`;
    tab.setAttribute('role', 'tab');
    tab.setAttribute('aria-label', example.name);
    tab.setAttribute('aria-controls', 'rubric-panel');
    tab.append(element('span', 'rubric-code', label), element('span', 'rubric-tab-name', example.name));
    tab.addEventListener('click', () => selectTab(index));
    parent.append(tab);
    return tab;
  });

  const selectTab = (index, moveFocus = false) => {
    const example = examples[index];
    tabs.forEach((tab, i) => {
      tab.setAttribute('aria-selected', String(i === index));
      tab.tabIndex = i === index ? 0 : -1;
    });
    if (moveFocus) tabs[index].focus();
    panel.setAttribute('aria-labelledby', tabs[index].id);
    const focus = element('div', 'rubric-focus');
    const media = element('div', 'rubric-media');
    const exampleLabel = example.paper_figure
      ? `Paper Figure ${String(example.paper_figure).replace(/^Figure\s+/i, '')}`
      : 'Benchmark example';
    media.append(videoPlayer(example, `${example.model} · ${example.name} example`),
      element('p', 'rubric-meta', `${example.model} · ${example.task_id} · ${exampleLabel}`));

    const description = element('div', 'rubric-text');
    description.append(element('div', 'media-label', `${example.id} · Score ${example.scores[example.id]} / 1`),
      element('h3', '', example.name), element('p', '', example.description),
      element('p', 'media-label', `WorldPhyReward reason · pass ${example.pass_id ?? 1}`),
      element('blockquote', '', example.reason), scoreList(example.scores));
    const task = element('details');
    task.append(element('summary', '', 'Initial frame and task instruction'));
    const caption = element('div', 'rubric-caption');
    caption.append(conditionImage(example.image, 'Task initial frame', true), element('p', '', example.caption));
    task.append(caption);
    description.append(task);
    focus.append(media, description);
    panel.querySelector('video')?.pause();
    panel.replaceChildren(focus);
  };
  parent.addEventListener('keydown', (event) => {
    const index = tabs.indexOf(event.target);
    if (index < 0) return;
    const offset = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 }[event.key];
    if (!offset && !['Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1
      : (index + offset + tabs.length) % tabs.length;
    selectTab(next, true);
  });
  selectTab(0);
}

function renderResults(models) {
  const select = byId('ranking');
  const body = byId('leaderboard-body');
  const paint = () => {
    const key = select.value;
    const showPassRate = key === 'passRate' || key === 'total';
    byId('metric-heading').textContent = showPassRate
      ? '7/7 pass rate'
      : `${shortNames[rubricIds.indexOf(key)]} mean / 1`;
    const measure = (model) => rubricIds.includes(key) ? model.scores[key] : model[key];
    const ordered = [...models].sort((a, b) => measure(b) - measure(a) || b.total - a.total);
    body.replaceChildren(...ordered.map((model, index) => {
      const row = element('tr');
      const barCell = element('td', 'bar-cell');
      const bar = element('div', 'metric-bar');
      const track = element('div', 'track');
      track.setAttribute('aria-hidden', 'true');
      const fill = element('span');
      fill.style.width = `${showPassRate ? model.passRate : model.scores[key] * 100}%`;
      track.append(fill);
      const value = showPassRate ? `${model.passRate.toFixed(2)}% ± ${model.passStd.toFixed(2)}` : model.scores[key].toFixed(3);
      bar.append(track, element('span', 'metric-value', value));
      barCell.append(bar);
      row.append(element('td', 'rank', index + 1), element('td', 'model', model.name),
        element('td', 'numeric', model.videos), barCell, element('td', 'numeric', model.total.toFixed(3)));
      return row;
    }));
  };
  select.addEventListener('change', paint);
  paint();
}

function renderTasks(data) {
  const groups = [
    ['End effector', 'gripper_hand'], ['Robot arms', 'arm_amount'], ['Camera view', 'camera_view'],
  ];
  byId('coverage').replaceChildren(...groups.map(([label, key]) => {
    const counts = Object.entries(data.coverage[key]).map(([name, count]) => `${name} ${count}`).join(' · ');
    const group = element('div', 'coverage-group');
    group.append(element('h3', '', label), element('p', '', counts));
    return group;
  }));
  byId('task-gallery').replaceChildren(...data.tasks.map((task) => {
    const card = element('article', 'task-card');
    const text = element('div', 'task-text');
    const details = element('details');
    details.append(element('summary', '', 'Read instruction'), element('p', '', task.caption));
    text.append(element('div', 'media-label', task.id),
      element('p', 'rubric-caption', Object.values(task.classification).join(' · ')), details);
    card.append(conditionImage(task.image, `Initial frame of task ${task.id}`, true), text);
    return card;
  }));
}

async function getData(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Cannot load ${path} (HTTP ${response.status})`);
  return response.json();
}

function showLoadError(ids, error) {
  ids.forEach((id) => byId(id).replaceChildren(element('p', 'load-error', 'Could not load this section. Reload the page to retry.')));
  console.error(error);
}

getData('data/demos.json').then((data) => {
  setupComparisons(data.comparisons);
  setupRubrics(data.rubrics);
}).catch((error) => showLoadError(['comparison', 'rubric-panel'], error));

getData('data/benchmark.json').then((data) => {
  renderResults(data.models);
  renderTasks(data);
}).catch((error) => {
  const row = element('tr');
  const cell = element('td', 'load-error', 'Could not load the results. Use the full results CSV above.');
  cell.colSpan = 5;
  row.append(cell);
  byId('leaderboard-body').replaceChildren(row);
  showLoadError(['task-gallery'], error);
});

byId('copy-citation').addEventListener('click', async (event) => {
  const button = event.currentTarget;
  try {
    await navigator.clipboard.writeText(byId('bibtex').textContent);
    button.textContent = 'Copied';
  } catch {
    const range = document.createRange();
    range.selectNodeContents(byId('bibtex'));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    button.textContent = 'Text selected';
  }
  window.setTimeout(() => { button.textContent = 'Copy BibTeX'; }, 1800);
});
