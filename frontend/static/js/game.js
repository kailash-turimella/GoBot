/**
 * game.js — Canvas rendering, click handling, and API calls for the 9×9 Go game.
 *
 * Board geometry:
 *   The 9×9 grid is drawn on a 540×540 canvas with 40px margins on each side.
 *   Cell spacing = (540 - 80) / 8 = 57.5 px.
 *   Stones are placed at grid INTERSECTIONS, not in cells.
 *   Intersection (row, col) maps to pixel (MARGIN + col*CELL, MARGIN + row*CELL).
 *
 * Click → intersection:
 *   nearest_col = round((x - MARGIN) / CELL)
 *   nearest_row = round((y - MARGIN) / CELL)
 *   Rejected if outside the board or farther than half a cell from the center.
 *
 * State model:
 *   A single `gameState` object mirrors the JSON shape returned by the API.
 *   Every API response fully replaces it; there is no local mutation.
 */

const BOARD_SIZE  = 9;
const CANVAS_SIZE = 540;
const MARGIN      = 40;
const CELL        = (CANVAS_SIZE - 2 * MARGIN) / (BOARD_SIZE - 1);   // 57.5
const STONE_R     = CELL * 0.44;

// Star-point (hoshi) positions for a 9×9 board
const STAR_POINTS = [[2,2],[2,6],[4,4],[6,2],[6,6]];
const COL_LABELS  = ['A','B','C','D','E','F','G','H','J']; // I skipped (standard Go)
const ROW_LABELS  = ['9','8','7','6','5','4','3','2','1']; // top→bottom = 9→1

let gameState = null;
let canvas, ctx;
let pollInterval = null;

// -----------------------------------------------------------------------
// Initialisation
// -----------------------------------------------------------------------

window.addEventListener('DOMContentLoaded', () => {
  canvas = document.getElementById('board-canvas');
  ctx    = canvas.getContext('2d');

  canvas.addEventListener('click', handleClick);
  document.getElementById('new-game-btn').addEventListener('click', newGame);
  document.getElementById('pass-btn').addEventListener('click', passTurn);
  document.getElementById('ai-btn').addEventListener('click', aiMove);
  document.getElementById('play-again-btn').addEventListener('click', () => {
    hideOverlay();
    newGame();
  });

  newGame();
  startPolling();
});

// -----------------------------------------------------------------------
// API calls
// -----------------------------------------------------------------------

async function newGame() {
  const res = await fetch('/new_game', { method: 'POST' });
  gameState  = await res.json();
  render();
}

async function handleClick(e) {
  if (gameState?.game_over) return;

  const rect = canvas.getBoundingClientRect();
  // Scale pixel coordinates to canvas logical coordinates
  const scaleX = CANVAS_SIZE / rect.width;
  const scaleY = CANVAS_SIZE / rect.height;
  const x = (e.clientX - rect.left) * scaleX;
  const y = (e.clientY - rect.top)  * scaleY;

  const intersection = pixelToIntersection(x, y);
  if (!intersection) return;
  const [row, col] = intersection;

  const res = await fetch('/move', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ row, col }),
  });

  if (res.ok) {
    gameState = await res.json();
    render();
  } else {
    const err = await res.json();
    showError(err.error || 'Illegal move');
  }
}

async function passTurn() {
  if (gameState?.game_over) return;
  const res = await fetch('/pass', { method: 'POST' });
  gameState  = await res.json();
  render();
}

async function aiMove() {
  if (gameState?.game_over) return;
  const btn = document.getElementById('ai-btn');
  btn.disabled = true;
  btn.textContent = 'Thinking…';
  try {
    const res = await fetch('/ai_move', { method: 'POST' });
    gameState  = await res.json();
    render();
  } finally {
    btn.disabled = false;
    btn.textContent = 'AI Move';
  }
}

// -----------------------------------------------------------------------
// Polling — sync board when the server state changes externally (e.g. AI via API)
// -----------------------------------------------------------------------

function startPolling() {
  clearInterval(pollInterval);
  pollInterval = setInterval(async () => {
    if (!gameState || gameState.game_over) return;
    try {
      const res = await fetch('/state');
      const fresh = await res.json();
      const lastMoveChanged = JSON.stringify(fresh.last_move) !== JSON.stringify(gameState.last_move);
      if (lastMoveChanged) {
        gameState = fresh;
        render();
      }
    } catch (_) {}
  }, 1000);
}

// -----------------------------------------------------------------------
// Coordinate helpers
// -----------------------------------------------------------------------

function intersectionToPixel(row, col) {
  return { x: MARGIN + col * CELL, y: MARGIN + row * CELL };
}

function pixelToIntersection(x, y) {
  const col = Math.round((x - MARGIN) / CELL);
  const row = Math.round((y - MARGIN) / CELL);
  if (col < 0 || col >= BOARD_SIZE || row < 0 || row >= BOARD_SIZE) return null;
  // Reject if the click is more than half a cell away from the nearest point
  const { x: px, y: py } = intersectionToPixel(row, col);
  if (Math.abs(x - px) > CELL * 0.5 || Math.abs(y - py) > CELL * 0.5) return null;
  return [row, col];
}

// -----------------------------------------------------------------------
// Rendering
// -----------------------------------------------------------------------

function render() {
  drawBoard();
  updateSidebar();
  if (gameState?.game_over) showOverlay();
}

function drawBoard() {
  // Wooden background
  ctx.fillStyle = '#dcb45b';
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

  // Subtle border/shadow inside canvas
  ctx.strokeStyle = '#b8903a';
  ctx.lineWidth = 3;
  ctx.strokeRect(MARGIN - 10, MARGIN - 10, CANVAS_SIZE - 2*(MARGIN-10), CANVAS_SIZE - 2*(MARGIN-10));

  // Grid lines
  ctx.strokeStyle = '#7a5c2a';
  ctx.lineWidth   = 1;
  for (let i = 0; i < BOARD_SIZE; i++) {
    const v = MARGIN + i * CELL;
    ctx.beginPath(); ctx.moveTo(v, MARGIN); ctx.lineTo(v, MARGIN + (BOARD_SIZE-1)*CELL); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(MARGIN, v); ctx.lineTo(MARGIN + (BOARD_SIZE-1)*CELL, v); ctx.stroke();
  }

  // Star points (hoshi)
  ctx.fillStyle = '#7a5c2a';
  for (const [r, c] of STAR_POINTS) {
    const { x, y } = intersectionToPixel(r, c);
    ctx.beginPath();
    ctx.arc(x, y, 3.5, 0, 2 * Math.PI);
    ctx.fill();
  }

  // Coordinate labels
  ctx.fillStyle = '#3a2a0a';
  ctx.font      = 'bold 13px sans-serif';

  // Column labels — top and bottom
  ctx.textAlign    = 'center';
  ctx.textBaseline = 'middle';
  for (let c = 0; c < BOARD_SIZE; c++) {
    const x = MARGIN + c * CELL;
    ctx.fillText(COL_LABELS[c], x, MARGIN / 2);
    ctx.fillText(COL_LABELS[c], x, CANVAS_SIZE - MARGIN / 2);
  }

  // Row labels — left and right
  ctx.textAlign = 'center';
  for (let r = 0; r < BOARD_SIZE; r++) {
    const y = MARGIN + r * CELL;
    ctx.fillText(ROW_LABELS[r], MARGIN / 2, y);
    ctx.fillText(ROW_LABELS[r], CANVAS_SIZE - MARGIN / 2, y);
  }

  if (!gameState) return;

  // Stones
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      const cell = gameState.board[r][c];
      if (cell === 0) continue;
      const { x, y } = intersectionToPixel(r, c);
      drawStone(x, y, cell === 1 ? 'black' : 'white');
    }
  }

  // Last-move indicator
  if (gameState.last_move) {
    const [r, c] = gameState.last_move;
    const { x, y } = intersectionToPixel(r, c);
    const isBlack = gameState.board[r][c] === 1;
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, 2 * Math.PI);
    ctx.fillStyle = isBlack ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.5)';
    ctx.fill();
  }
}

function drawStone(x, y, color) {
  ctx.beginPath();
  ctx.arc(x, y, STONE_R, 0, 2 * Math.PI);

  if (color === 'black') {
    const grad = ctx.createRadialGradient(x - STONE_R*0.3, y - STONE_R*0.3, STONE_R*0.1,
                                           x, y, STONE_R);
    grad.addColorStop(0, '#6a6a6a');
    grad.addColorStop(1, '#0a0a0a');
    ctx.fillStyle = grad;
  } else {
    const grad = ctx.createRadialGradient(x - STONE_R*0.3, y - STONE_R*0.3, STONE_R*0.05,
                                           x, y, STONE_R);
    grad.addColorStop(0, '#ffffff');
    grad.addColorStop(1, '#c8c8c8');
    ctx.fillStyle = grad;
    ctx.strokeStyle = '#909090';
    ctx.lineWidth = 1;
    ctx.stroke();
  }
  ctx.fill();
}

// -----------------------------------------------------------------------
// Sidebar updates
// -----------------------------------------------------------------------

function updateSidebar() {
  if (!gameState) return;

  const turnStone  = document.getElementById('turn-stone');
  const turnLabel  = document.getElementById('turn-label');
  const blackCap   = document.getElementById('black-captures');
  const whiteCap   = document.getElementById('white-captures');
  const scoresBlk  = document.getElementById('scores-block');

  blackCap.textContent = gameState.captured.black;
  whiteCap.textContent = gameState.captured.white;

  if (gameState.game_over) {
    turnLabel.textContent = 'Game Over';
    turnStone.className = '';
    if (gameState.scores) {
      document.getElementById('score-black').textContent = gameState.scores.black;
      document.getElementById('score-white').textContent = gameState.scores.white;
      scoresBlk.classList.remove('hidden');
    }
  } else {
    const isBlack = gameState.turn === 'black';
    turnLabel.textContent = isBlack ? "Black's turn" : "White's turn";
    turnStone.className   = isBlack ? '' : 'white';
    scoresBlk.classList.add('hidden');
  }
}

// -----------------------------------------------------------------------
// Game-over overlay
// -----------------------------------------------------------------------

function showOverlay() {
  const winner = capitalize(gameState.winner);
  document.getElementById('winner-text').textContent = `${winner} wins!`;
  if (gameState.scores) {
    document.getElementById('final-scores-text').innerHTML =
      `Black: ${gameState.scores.black}<br>White: ${gameState.scores.white} (incl. 2.5 komi)`;
  }
  document.getElementById('overlay').classList.remove('hidden');
}

function hideOverlay() {
  document.getElementById('overlay').classList.add('hidden');
}

// -----------------------------------------------------------------------
// Error flash
// -----------------------------------------------------------------------

let errorTimer = null;

function showError(msg) {
  const el = document.getElementById('error-flash');
  el.textContent = msg;
  el.classList.remove('hidden');
  clearTimeout(errorTimer);
  errorTimer = setTimeout(() => el.classList.add('hidden'), 2200);
}

// -----------------------------------------------------------------------
// Utility
// -----------------------------------------------------------------------

function capitalize(str) {
  return str ? str.charAt(0).toUpperCase() + str.slice(1) : '';
}
