CREATE TABLE sessions (
  id TEXT PRIMARY KEY,
  document TEXT NOT NULL,
  filename TEXT NOT NULL DEFAULT 'Welcome guide',
  chunks TEXT NOT NULL DEFAULT '[]',
  graph TEXT,
  agent TEXT,
  history TEXT NOT NULL DEFAULT '[]',
  revision INTEGER NOT NULL DEFAULT 0,
  updated_at INTEGER NOT NULL
);
CREATE INDEX sessions_expiry ON sessions(updated_at);
CREATE TABLE rate_limits (
  key TEXT PRIMARY KEY,
  count INTEGER NOT NULL,
  expires_at INTEGER NOT NULL
);
CREATE INDEX rate_limits_expiry ON rate_limits(expires_at);
