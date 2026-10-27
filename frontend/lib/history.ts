import { useSyncExternalStore } from "react";
import type { DebugResponse, Framework } from "./types";

export type HistoryEntry = {
  id: string;
  at: number;
  question: string;
  framework: Framework | null;
  result: DebugResponse;
};

const KEY = "neural-debugger:history:v1";
const EVENT = "neural-debugger:history";
export const MAX_ENTRIES = 10;
const EMPTY: HistoryEntry[] = [];

function storage(): Storage | null {
  try {
    return typeof window === "undefined" ? null : window.localStorage;
  } catch {
    return null; // blocked / private mode
  }
}

function readRaw(): string | null {
  try {
    return storage()?.getItem(KEY) ?? null;
  } catch {
    return null;
  }
}

export function parseHistory(raw: string | null): HistoryEntry[] {
  if (!raw) return EMPTY;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return EMPTY;
    return (parsed as HistoryEntry[]).filter((e) => typeof e?.question === "string" && typeof e?.result === "object");
  } catch {
    return EMPTY;
  }
}

export function loadHistory(): HistoryEntry[] {
  return parseHistory(readRaw());
}

function notify() {
  if (typeof window !== "undefined") window.dispatchEvent(new Event(EVENT));
}

export function saveHistory(entries: HistoryEntry[]): void {
  try {
    storage()?.setItem(KEY, JSON.stringify(entries.slice(0, MAX_ENTRIES)));
  } catch {
    /* quota or blocked storage: history is a convenience, never required */
  }
  notify();
}

export function addEntry(entries: HistoryEntry[], entry: HistoryEntry): HistoryEntry[] {
  return [entry, ...entries.filter((e) => e.question !== entry.question)].slice(0, MAX_ENTRIES);
}

export function clearHistory(): void {
  try {
    storage()?.removeItem(KEY);
  } catch {
    /* ignore */
  }
  notify();
}

let cachedRaw: string | null | undefined;
let cachedValue: HistoryEntry[] = EMPTY;

function getSnapshot(): HistoryEntry[] {
  const raw = readRaw();
  if (raw !== cachedRaw) {
    cachedRaw = raw;
    cachedValue = parseHistory(raw);
  }
  return cachedValue;
}

function subscribe(onChange: () => void): () => void {
  window.addEventListener(EVENT, onChange);
  window.addEventListener("storage", onChange); // other tabs
  return () => {
    window.removeEventListener(EVENT, onChange);
    window.removeEventListener("storage", onChange);
  };
}

/** The saved history, empty on the server and during hydration, live afterwards. */
export function useHistory(): HistoryEntry[] {
  return useSyncExternalStore(subscribe, getSnapshot, () => EMPTY);
}
