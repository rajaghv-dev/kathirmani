-- Rollback 0001_init. Drops in dependency order.
DROP TABLE IF EXISTS job_queue;
DROP TABLE IF EXISTS model_benchmark_runs;
DROP TABLE IF EXISTS model_runs;
DROP TABLE IF EXISTS model_registry;
DROP TABLE IF EXISTS model_profiles;
DROP TABLE IF EXISTS embeddings;
DROP TABLE IF EXISTS incident_events;
DROP TABLE IF EXISTS incidents;
DROP TABLE IF EXISTS vlm_observations;
DROP TABLE IF EXISTS events;          -- drops partitions (events_default) too
DROP TABLE IF EXISTS tracks;
DROP TABLE IF EXISTS detections;
DROP TABLE IF EXISTS ai_windows;
DROP TABLE IF EXISTS video_segments;
DROP TABLE IF EXISTS zones;
DROP TABLE IF EXISTS cameras;
DROP TABLE IF EXISTS stores;
