/** Periodic polling hook — replaces Gradio's gr.Timer for real-time updates. */

import { useState, useEffect, useRef, useCallback } from 'react';
import { getRealtimeSnapshot } from '../api/client';
import type { RealtimeSnapshot } from '../api/types';

export function usePolling(taskId: string, intervalMs = 5000) {
  const [data, setData] = useState<RealtimeSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchSnapshot = useCallback(async () => {
    try {
      const snapshot = await getRealtimeSnapshot(taskId);
      setData(snapshot);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    }
  }, [taskId]);

  useEffect(() => {
    fetchSnapshot();
    timerRef.current = setInterval(fetchSnapshot, intervalMs);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [fetchSnapshot, intervalMs]);

  return { data, error, refresh: fetchSnapshot };
}
