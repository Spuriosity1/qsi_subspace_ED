import io
import numpy as np
from db import connect_npsql, convert_array, get_sector_list
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from Jring_data import Jring

def convert_sparray(txt):
    out = io.BytesIO(txt)
    out.seek(0)
    return np.load(out,allow_pickle=True)


def num_cells(lat):
    if type(lat) in [bytes, str]:
        lat = convert_sparray(lat)
    return int(np.rint(abs(np.linalg.det(np.array(lat,dtype=np.float64)))/128))


def load_geometries(con):
    curs = con.cursor()
    curs.execute("SELECT latvecs FROM field_111 GROUP BY latvecs")
    lats = [x[0] for x in curs.fetchall()]
    curs.close()
    return lats


def find_groundstate_impl2(con, lat, sign, x_name, table_name, sign_name):
    c = con.cursor()
    x_list = np.array(c.execute(f"""
        SELECT {x_name} FROM {table_name}
        WHERE latvecs = ? AND {sign_name} = ? 
        GROUP BY {x_name}
        ORDER BY {x_name}
        """,
        (lat, sign)).fetchall())[:,0]

    for x in x_list:

    # best_sectors = []
    # sector_energies = []
    
    # for x in x_list:
    #     res = c.execute(f"""
    #         SELECT edata, sector FROM {table_name}
    #         WHERE {x_name} = ? AND latvecs = ? AND {sign_name} = ?""",
    #                     (x,lat, sign) )
    #     gse = np.inf
    #     best_sec = None
        
    #     for e, sec in res:
    #         curr_gse = convert_array(e)[0]
    #         if  curr_gse < gse:
    #             gse = curr_gse
    #             best_sec = sec
                
        
        
            
        


    # c.close()


def find_groundstate_impl(con, lat, sign, x_name, table_name, sign_name):
    c = con.cursor()
    x_list = np.array(c.execute(f"""
        SELECT {x_name} FROM {table_name}
        WHERE latvecs = ? AND {sign_name} = ? 
        GROUP BY {x_name}
        ORDER BY {x_name}
        """,
        (lat, sign)).fetchall())[:,0]

    # best_sectors = []
    # sector_energies = []
    
    # for x in x_list:
    #     res = c.execute(f"""
    #         SELECT edata, sector FROM {table_name}
    #         WHERE {x_name} = ? AND latvecs = ? AND {sign_name} = ?""",
    #                     (x,lat, sign) )
    #     gse = np.inf
    #     best_sec = None
        
    #     for e, sec in res:
    #         curr_gse = convert_array(e)[0]
    #         if  curr_gse < gse:
    #             gse = curr_gse
    #             best_sec = sec
                
        
        
            
        


    # c.close()


    sectors = list(get_sector_list(con, lat).keys())

    energy_list = []
    
    for sector in sectors:
        
        
        res = c.execute(f"""
        SELECT {x_name}, edata FROM {table_name}
        WHERE sector = ? AND latvecs = ? AND {sign_name} = ? 
        ORDER BY {x_name}
        """,
        (sector, lat, sign))
        
        tmp = []
        for x, e in res:
            e = convert_array(e)[0]
            tmp.append(e)

        assert len(tmp) == len(x_list), f"sector {sector} - Lengths don't match - {len(tmp)} != {len(x_list)}"
        energy_list.append(tmp)

    best_sectors = []
    
    
    energy_list = np.array(energy_list).T

    sector_energies = []
    for x, E_set in zip(x_list, energy_list):
        best_sector_idx = np.argsort(E_set)[:5]
        best_sectors.append(best_sector_idx)
        sector_energies.append(E_set[best_sector_idx])

    c.close()
    return x_list, best_sectors, sector_energies, sectors
        
        
def find_groundstate_111(con, lat, sign=1):
    return find_groundstate_impl(con, lat, sign, 
                          x_name='g0_g123',
                          table_name='field_111',
                          sign_name='g123_sign'
                         )


def find_groundstate_110(con, lat, sign=1):
    return find_groundstate_impl(con, lat, sign, 
                          x_name='g01_g23',
                          table_name='field_110',
                          sign_name='g23_sign'
                         )

def process_expO(raw):
    return [
        [np.mean(np.real(np.linalg.eigvals(O[sl]))) for O in raw]
        for sl in range(4)
    ]
    


class RingInterpolator:
    def __init__(self, data_importer, interpolation_f=CubicSpline, n_energies=5):
        
        x_list_plus, E_list_plus, expO_list_plus = data_importer(1)
        expO_series_plus = process_expO(expO_list_plus)
    
        x_list_minus, E_list_minus, expO_list_minus = data_importer(-1)
        expO_series_minus = process_expO(expO_list_minus)

    
        mask_plus = np.diff(np.hstack((x_list_plus,np.inf))) > 1e-10
        mask_minus = np.diff(np.hstack((x_list_minus,np.inf))) > 1e-10

        print(mask_plus, x_list_plus, len(mask_plus), len(x_list_plus))

        x_list_plus =np.array( x_list_plus)[mask_plus]
        E_list_plus =np.array( E_list_plus)[mask_plus, :]

        x_list_minus = np.array(x_list_minus)[mask_minus]
        E_list_minus = np.array(E_list_minus)[mask_minus, :]
        
        for sl in range(4):
            expO_series_minus[sl] = np.array(expO_series_minus[sl])[mask_minus]
            expO_series_plus[sl] = np.array(expO_series_plus[sl])[mask_plus]

        

        E_list_plus = np.sort(E_list_plus, axis=-1)
        E_list_minus = np.sort(E_list_minus, axis=-1)
    
        self.ring_interpolators = {
             1: [ interpolation_f(x_list_plus, expO_series_plus[sl]) for sl in range(4)],
            -1: [ interpolation_f(x_list_minus, expO_series_minus[sl]) for sl in range(4)]
        }

        self.E_interpolators = [{
             1 : interpolation_f(x_list_plus, np.array(E_list_plus)[:,n]),
            -1 : interpolation_f(x_list_minus, np.array(E_list_minus)[:,n])
        } for n in range(n_energies)]

        self.x_list = {1: x_list_plus, -1:x_list_minus}
        self.E_list = {1: E_list_plus, -1:E_list_minus}
        self.expO_series = {1: expO_series_plus, -1: expO_series_minus}
    
        
            
    def interpolate_ring(self, sign, x, check=True):
        if x > np.max(self.x_list[sign]):
            if check:
                return [np.nan, np.nan, np.nan, np.nan]
            return self.expO_series[-1]
        elif x < np.min(self.x_list[sign]):
            if check:
                return [np.nan, np.nan, np.nan, np.nan]
            return self.expO_series[0]
            
        return [self.ring_interpolators[sign][j](x) for j in range(4)]

    def check_g_compatible(self, g):
        raise NotImplementedError()
    
    def O(self, g, check=True):
        self.check_g_compatible(g)
        x = g[0] / g[3]
        
        if g[3] >= 0:
            return self.interpolate_ring(1, x, check)
        else:
            return self.interpolate_ring(-1, x, check)
            
    def gap(self, g, check=True):
        self.check_g_compatible(g)
        
        x = g[0]/ g[3]
        
        sign = 1 if g[3] >= 0 else -1
        
        e0 = self.E_interpolators[0][sign](x)
        e1 = self.E_interpolators[1][sign](x)

        if x > np.max(self.x_list[sign]):
            # if check:
            return np.nan
            
        elif x < np.min(self.x_list[sign]):
            # if check:
            return np.nan
            
        
        # rescale by g[3]
        e1 *= np.abs(g[3])
        e0 *= np.abs(g[3])
        return e1 - e0


def close(x,y,tol=1e-10):
    return np.abs(x-y)<tol


def load_E_and_expO(res):
    x_list = []
    E_list = []
    expO_list = []
    
    tmp = res.fetchone()
    
    while tmp is not None: 
        x_list.append(tmp[0])
        E_list.append(convert_array(tmp[1]))
        expO_list.append([convert_array(x) for x in tmp[2:]])
        tmp = res.fetchone()
    
    E_list = np.array(E_list)

    return x_list, E_list, expO_list
    




def calc_phasedia_data(rfi: RingInterpolator, field_direction, 
                       resolution = (128,64), 
                       jpm_limits=(-0.1,0.1),
                       B_limits =(0,0.5)
                      ):
    
    jpm_arr = np.linspace(*jpm_limits,resolution[0])
    B_arr   = np.linspace(*B_limits, resolution[1])

    rf_vals = np.empty((4,resolution[1],resolution[0]))
    gap_vals = np.empty((resolution[1],resolution[0]))
    x_vals = np.empty((resolution[1],resolution[0]))

    rf_vals[:] = np.nan
    gap_vals[:] = np.nan

    field_direction /= np.linalg.norm(field_direction)
    
    for i, jpm in enumerate(jpm_arr):
        for j, b in enumerate(B_arr):
            g = Jring(jpm, b*field_direction)
            rf_vals[:,j,i] = rfi.O( g, check=True )
            gap_vals[j,i] = rfi.gap(g, check=True)
            x_vals[j, i] = g[0]/g[3]
            
    return jpm_arr, B_arr, rf_vals, gap_vals, x_vals
