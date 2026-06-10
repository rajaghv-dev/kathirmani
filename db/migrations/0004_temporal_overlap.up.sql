-- spec/10 "Temporal correlation" — enable TRUE cross-clip / cross-camera overlap
-- joins for long or subtle actions (palming 3s, tag-swap 12s, sweethearting 35s)
-- that the relative per-window sec-offsets + plain btree indexes cannot express.
--
-- We add an absolute `tstzrange` span to the extent-bearing rows and GiST overlap
-- indexes. btree_gist lets us put a scalar key (camera_id / store_id) in the SAME
-- GiST index, so "rows on camera X whose span && :range" is fully index-served.
CREATE EXTENSION IF NOT EXISTS btree_gist;

-- ---- video_segments: start/end are NOT NULL → a STORED generated span ----------
ALTER TABLE video_segments
  ADD COLUMN span tstzrange
  GENERATED ALWAYS AS (tstzrange(start_time, end_time, '[)')) STORED;
CREATE INDEX idx_segments_cam_span ON video_segments USING gist (camera_id, span);

-- ---- incidents: start/end are nullable → span only when both are known ---------
ALTER TABLE incidents
  ADD COLUMN span tstzrange
  GENERATED ALWAYS AS (
    CASE WHEN start_time IS NOT NULL AND end_time IS NOT NULL
         THEN tstzrange(start_time, end_time, '[]') END
  ) STORED;
CREATE INDEX idx_incidents_store_span ON incidents USING gist (store_id, span);

-- ---- ai_windows: absolute time = parent segment.start_time + the sec offsets ----
-- A generated column can't reference another table, so we keep a plain `span` and
-- maintain it with a trigger (self-maintaining → no app change needed) + backfill.
ALTER TABLE ai_windows ADD COLUMN span tstzrange;

CREATE OR REPLACE FUNCTION ai_windows_set_span() RETURNS trigger AS $$
DECLARE seg_start TIMESTAMPTZ;
BEGIN
  SELECT start_time INTO seg_start FROM video_segments WHERE id = NEW.segment_id;
  IF seg_start IS NOT NULL THEN
    NEW.span := tstzrange(
      seg_start + make_interval(secs => NEW.window_start_sec::double precision),
      seg_start + make_interval(secs => NEW.window_end_sec::double precision),
      '[)');
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_ai_windows_span
  BEFORE INSERT OR UPDATE OF segment_id, window_start_sec, window_end_sec
  ON ai_windows FOR EACH ROW EXECUTE FUNCTION ai_windows_set_span();

-- backfill existing rows
UPDATE ai_windows w SET span = tstzrange(
    s.start_time + make_interval(secs => w.window_start_sec::double precision),
    s.start_time + make_interval(secs => w.window_end_sec::double precision),
    '[)')
  FROM video_segments s WHERE w.segment_id = s.id;

CREATE INDEX idx_windows_cam_span ON ai_windows USING gist (camera_id, span);

-- NOTE: `events` is point-in-time (event_time) and already has idx_events_type_time;
-- "events within [a,b]" is a btree range scan, so it needs no GiST span here.
