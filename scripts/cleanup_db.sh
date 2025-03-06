#!/bin/bash

# Set database name and table
DB="$1"


TABLE="field_111"
COL1="g0_g123"
COL2="g123_sign"

sqlite3 "$DB" <<EOF
DELETE FROM $TABLE
WHERE EXISTS (
    SELECT 1 FROM $TABLE AS t2
	WHERE $TABLE.$COL1 BETWEEN (t2.$COL1 - 0.000001) AND (t2.$COL1 + 0.000001)
    AND $TABLE.$COL2 = t2.$COL2
	AND $TABLE.latvecs = t2.latvecs
    AND $TABLE.rowid > t2.rowid
);
EOF


echo "De-duplication complete!"
