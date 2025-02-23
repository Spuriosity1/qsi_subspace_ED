#!/usr/bin/env bash
# usage: sqlite-merge-dbs out.sqlite in0.sqlite in1.sqlite in2.sqlite ...

set -eu
outdb="$1"
shift
indb0="$1"
shift
cp "$indb0" "$outdb"
for table in $(sqlite3 "$outdb" "SELECT name FROM sqlite_master WHERE type='table'"); do
  echo "table: $table"
  for db in "$@"; do
    echo "db: $db"
    sqlite3 "$outdb" <<EOF
      ATTACH '$db' AS db2;
      INSERT OR IGNORE INTO "$table" SELECT * FROM db2."$table";
      DETACH db2;
EOF
  done
done
