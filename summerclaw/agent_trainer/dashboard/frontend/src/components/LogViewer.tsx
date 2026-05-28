/** Log viewer — terminal-style scrollable log display. */

import React, { useRef, useEffect } from 'react';

interface Props {
  lines: string[];
  maxHeight?: number;
}

export const LogViewer: React.FC<Props> = ({ lines, maxHeight = 500 }) => {
  const containerRef = useRef<HTMLPreElement>(null);

  // Auto-scroll to bottom when new lines arrive
  useEffect(() => {
    const el = containerRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [lines]);

  return (
    <pre
      ref={containerRef}
      style={{
        background: '#1e1e1e',
        color: '#d4d4d4',
        padding: 16,
        borderRadius: 6,
        maxHeight,
        overflow: 'auto',
        fontSize: 13,
        fontFamily: "'Fira Code', 'Consolas', 'Monaco', monospace",
        lineHeight: 1.5,
        margin: 0,
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-all',
      }}
    >
      {lines.length > 0 ? lines.join('\n') : 'No logs yet...'}
    </pre>
  );
};
