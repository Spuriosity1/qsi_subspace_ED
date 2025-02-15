from db import connect_npsql, convert_array, get_sector_list
import numpy as np

import sys

if len(sys.argv) < 2:
    print(f"USAGE: {sys.argv[0]} path_to_database")



def _cleanup_impl(x_name, sign_colname, table_name, lat, sector):
    c= con.cursor()
    for sign in [-1,1]:
        c.execute(f"""
            SELECT {x_name}
            FROM {table_name}
            WHERE latvecs = ? AND sector = ? AND {sign_colname} = ?
            ORDER BY {x_name}
            """,(lat,sector, sign))
        
        res = c.fetchall()
        if len(res) == 0:
            continue
        problems = np.argwhere(np.diff(np.array(res)[:,0]) < 1e-6)
        for J in problems:
            x = res[J[0]][0]
            print(f"Deleting {x_name} = {x}")
            q = f"""
                DELETE FROM {table_name} WHERE rowid = (
                    SELECT rowid FROM {table_name}
                    WHERE {x_name} = ? AND latvecs = ? AND sector = ? AND {sign_colname} = ?
                    LIMIT 1
                )
                """
            c.execute(q, (x,lat,sector,sign)   )




con = connect_npsql(sys.argv[1])


curs = con.cursor()

curs.execute("SELECT latvecs FROM field_111 GROUP BY latvecs")
lats = [x[0] for x in curs.fetchall()]
curs.close()

def cleanup_111(lat):
    for s in get_sector_list(con, lat).keys():
        _cleanup_impl('g0_g123', 'g123_sign', 'field_111', lat, s)


def cleanup_110(lat):
    for s in get_sector_list(con, lat).keys():
        _cleanup_impl('g01_g23', 'g23_sign', 'field_110', lat, s)



for l in lats:
    cleanup_111(l)
    cleanup_110(l)


con.commit()
con.close()
