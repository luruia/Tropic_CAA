
; undef to_season(var)
function to_season(var)     ;;;   返回季节平均的距平
local dims, n_year, temp, var_season
begin
    
    dims = dimsizes(var)
    n_year = dims(0)/12

    temp = reshape(var, (/n_year, 12, dims(1), dims(2)/))
    var_season = dim_avg_n(temp(:,5:8,:,:), 1)
    copy_VarCoords(var(0,:,:), var_season(0,:,:))
    var_season!0 = "time"
    var_season&time = ispan(1, n_year, 1)
    var_season = dtrend_msg_n(var_season&time, var_season, True, False, 0)

    return var_season
end

;;;>-----------------------------------------------------------------------------------------<


begin

;;;>-----------------------------------------------------------------------------------------<
;;;                                时间段
;;;>-----------------------------------------------------------------------------------------<

yyyymm_s = 197901
yyyymm_e = 202112

;;;>-----------------------------------------------------------------------------------------<
;;;                               读取数据
;;;>-----------------------------------------------------------------------------------------<

f_pc_olr = addfile("./olr_index.nc", "r")
pc_olr = f_pc_olr->pc(yyyymm_s/100-1979:yyyymm_e/100-1979)
pc_olr = dim_standardize(pc_olr, 1)
pc_olr = pc_olr * (-1.)

;;;>-----------------------------------------------------------------------------------------<

f3 = addfile("./ERA5_2mTSLP_1x1_1940_2022_mon.nc", "r")
time = cd_calendar(f3->time, 1) 
ind_s = ind(time .eq. yyyymm_s)
ind_e = ind(time .eq. yyyymm_e)
t2m = short2flt(f3->t2m(ind_s:ind_e,:,:))
dims_t2m = dimsizes(t2m)

t2m_anom = to_season(t2m)
delete(t2m)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

lonw1 = 230
lone1 = 280
lats1 = 72
latn1 = 80

t2m_index = dim_avg_n_Wrap(t2m_anom(:,{lats1:latn1},{lonw1:lone1}), (/1, 2/))
t2m_index = dim_standardize(t2m_index, 1)

ff = addfile("./t2m_index.nc", "c")
ff->pc = t2m_index

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<


end
