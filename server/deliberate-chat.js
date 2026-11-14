// Deliberative response selection and honest word-paced streaming.
// AXON still generates every draft using its own trained checkpoint.
// This module only ranks generated drafts; it does not write answers.
const TERMS = /[\p{L}\p{N}][\p{L}\p{N}'’-]*/gu;
const WORDS = /\s*\S+|\s+/gu;
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

function terms(text) {
  return (String(text || '').toLowerCase().match(TERMS) || []).filter(word => word.length > 3);
}

export function draftScore(draft, messages=[]) {
  if (typeof draft !== 'string' || !draft.trim()) return -Infinity;
  const trimmed = draft.trim(), words = terms(trimmed), lower = trimmed.toLowerCase();
  const lastQuestion = [...messages].reverse().find(item => item.role === 'user')?.content || '';
  const important = new Set(terms(lastQuestion));
  const matched = new Set(words.filter(word => important.has(word)));
  const priorReplies = messages.filter(item => item.role === 'assistant').slice(-2)
    .map(item => item.content.toLowerCase());
  let score = Math.min(words.length, 25) * 0.10 + Math.min(matched.size, 5) * 0.35;
  if (/[.!?…]$/u.test(trimmed)) score += 0.65;
  if (/^[\p{Lu}\p{N}]/u.test(trimmed)) score += 0.15;
  if (/[\u0000-\u001f\u007f\ufffd]/u.test(trimmed)) score -= 5;
  if (/(.)\1{4,}/u.test(trimmed)) score -= 4;
  if (words.length > 6 && new Set(words).size / words.length < .48) score -= 4;
  if (priorReplies.some(reply => reply.includes(lower.slice(0, Math.min(35,lower.length))))) score -= 3;
  // Penalize repeated phrases. This is a quality heuristic, not simulated reasoning.
  const triples = new Set();
  for (let i=0; i+2<words.length; i++) {
    const phrase=words.slice(i,i+3).join(' ');
    if (triples.has(phrase)) score -= 0.8;
    triples.add(phrase);
  }
  return score;
}

export function chooseDraft(drafts, messages=[], {accept=()=>true}={}) {
  return drafts.filter(draft => typeof draft === 'string' && draft.trim() && accept(draft))
    .map(draft => draft.trim())
    .sort((a,b) => draftScore(b,messages)-draftScore(a,messages))[0] || '';
}

// A "word" here is a non-whitespace group, preserving leading spaces and punctuation.
// Each visible word must wait at least 500 ms. Time spent waiting is not neural thinking.
export async function* pacedWords(text, {minimumMs=500, wait=pause}={}) {
  for (const part of String(text).match(WORDS) || []) {
    if (part.trim()) await wait(Math.max(500, minimumMs));
    yield part;
  }
}

export function countWords(text) {
  return (String(text).match(/\S+/gu) || []).length;
}
