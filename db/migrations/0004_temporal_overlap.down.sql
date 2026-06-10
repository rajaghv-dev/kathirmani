-- Roll back the temporal-overlap spans/indexes. We leave the btree_gist extension
-- installed (cheap, possibly used elsewhere).
DROP INDEX IF EXISTS idx_windows_cam_span;
DROP TRIGGER IF EXISTS trg_ai_windows_span ON ai_windows;
DROP FUNCTION IF EXISTS ai_windows_set_span();
ALTER TABLE ai_windows DROP COLUMN IF EXISTS span;

DROP INDEX IF EXISTS idx_incidents_store_span;
ALTER TABLE incidents DROP COLUMN IF EXISTS span;

DROP INDEX IF EXISTS idx_segments_cam_span;
ALTER TABLE video_segments DROP COLUMN IF EXISTS span;
