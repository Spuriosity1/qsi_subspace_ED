import sqlite3
import io
import numpy as np


def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)


def connect_npsql(db_path, **kwargs):
    return sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES, **kwargs)


def get_sector_list(con, lat):
    retval = {}
    res = con.execute("""
        SELECT sector, count(sector)
        FROM field_111
        WHERE latvecs = ?
        GROUP BY sector
        """, (lat,))
    for sector, count in res:
        retval[sector] = count
    res.close()
        
    return retval


def init_db(con):
    con.execute("""
        create table field_110 ( g01_g23 REAL, g23_sign INTEGER,
                edata BLOB,
                reO0 BLOB, reO1 BLOB, reO2 BLOB, reO3 BLOB,
                latvecs BLOB, sector BLOB, kx FLOAT, ky FLOAT, kz FLOAT
                )
                """)
    con.execute("""
create table field_111 ( g0_g123 REAL, g123_sign INTEGER,
                        edata BLOB,
                reO0 BLOB, reO1 BLOB, reO2 BLOB, reO3 BLOB,
                latvecs BLOB, sector BLOB, kx FLOAT, ky FLOAT, kz FLOAT
                        )
                """)



rotation_matrices = {
    'I': np.array([[+1, 0, 0], [0, +1, 0], [0, 0, +1]]),
    'X': np.array([[+1, 0, 0], [0, -1, 0], [0, 0, -1]]),
    'Y': np.array([[-1, 0, 0], [0, +1, 0], [0, 0, -1]]),
    'Z': np.array([[-1, 0, 0], [0, -1, 0], [0, 0, +1]])
        }
