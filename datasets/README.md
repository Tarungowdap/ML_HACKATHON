# Datasets (schema 1.0)
- processed/  observations_*, candidates_*, arcs_*, nodes_*, routes_* (parquet). Key: (trip_id, obs_idx[, arc_id]). arc_id/node_id are PER TRIP.
- reports/    data audit, per-trip miss rates, recall vs radius (see data_report.md)
- sample/     3-trip sample for quick tests
- schema.json, splits.json, columns_dictionary.csv, CHANGELOG.md

Large files (not in git): arcs_train.parquet, nodes_train.parquet. Download them into datasets/processed/:
    pip install gdown
    gdown --folder https://drive.google.com/drive/folders/1342ob9YBsZM3m34dyDkIlpmidG1kusfZ -O datasets/processed_download
then copy those two files into datasets/processed/.
candidates_train.parquet is stored with Git LFS.

Loader: src/pathsnap/pathsnap.py -> load_trip(trip_id, data_dir="datasets/processed")
