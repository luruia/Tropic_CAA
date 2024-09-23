load "$NCARG_ROOT/lib/ncarg/nclscripts/contrib/run_cor.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

;;;>-----------------------------------------------------------------------------------------<

; undef runave_lr(var, nave)
function runave_lr(var, nave)     ;;;   分离空间场[time; lat; lon]的年际、年代际部分。
local var_mean, var_year, m, dims
begin
    dims = dimsizes(var)
    var_mean = runave_n_Wrap(var, nave, 0, 0)

    do m = 0, (nave/2 - 1), 1
        var_mean(m,:,:) = dim_avg_n_Wrap(var(0:nave/2+m,:,:), 0)
        var_mean(dims(0)-1-m,:,:) = dim_avg_n_Wrap(var(dims(0)-nave/2 -1 -m:,:,:), 0)
    end do

    var_year = var - var_mean
    copy_VarMeta(var, var_year)
    copy_VarMeta(var, var_mean)

    return [/var_mean, var_year/]
end

;;;>-----------------------------------------------------------------------------------------<

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


; undef to_clim(var)
function to_clim(var)     ;;;   返回季节平均的气候态
local dims, n_year, temp, var_season, var_clim
begin
    
    dims = dimsizes(var)
    n_year = dims(0)/12

    temp = reshape(var, (/n_year, 12, dims(1), dims(2)/))
    var_season = dim_avg_n(temp(:,5:8,:,:), 1)
    copy_VarCoords(var(0,:,:), var_season(0,:,:))
    var_season!0 = "time"
    var_season&time = ispan(1, n_year, 1)
    var_season = dtrend_msg_n(var_season&time, var_season, False, False, 0)
    var_clim = dim_avg_n_Wrap(var_season, 0)

    return var_clim
end

;;;>-----------------------------------------------------------------------------------------<


begin

;;;>-----------------------------------------------------------------------------------------<
;;;                                时间段
;;;>-----------------------------------------------------------------------------------------<


yyyymm_s = 197901
yyyymm_e = 202112


;;;   是否对环流场进行去年代际信号处理
rm_id_circ = False

;;;>-----------------------------------------------------------------------------------------<
;;;                               读取数据
;;;>-----------------------------------------------------------------------------------------<

f_sst = addfile("./sst_index_polar.nc", "r")
pc_sst = f_sst->pc
delete(f_sst)

pc = dim_standardize(pc_sst, 1) * (-1.)

;;;>-----------------------------------------------------------------------------------------<

idir = "./era5/monthly/"

f1 = addfile(idir + "ERA5_Z_1x1_1940_2023_mon.nc", "r") 
time = cd_calendar(f1->time, 1) 
ind_s = ind(time .eq. yyyymm_s)
ind_e = ind(time .eq. yyyymm_e)
hgt_200 = short2flt(f1->z(ind_s:ind_e,{250},:,:))
hgt_200 = hgt_200/9.8
lon = f1->longitude
lat = f1->latitude
nlon = dimsizes(lon)
nlat = dimsizes(lat)
dims_uvz = dimsizes(hgt_200)
delete([/time, ind_s, ind_e, f1/])

f2 = addfile(idir + "ERA5_U_1x1_1940_2023_mon.nc", "r")
time = cd_calendar(f2->time, 1) 
ind_s = ind(time .eq. yyyymm_s)
ind_e = ind(time .eq. yyyymm_e)
u_200 = short2flt(f2->u(ind_s:ind_e,{250},:,:)) 
delete([/time, ind_s, ind_e, f2/])

f3 = addfile(idir + "ERA5_V_1x1_1940_2023_mon.nc", "r")  
time = cd_calendar(f3->time, 1) 
ind_s = ind(time .eq. yyyymm_s)
ind_e = ind(time .eq. yyyymm_e)
v_200 = short2flt(f3->v(ind_s:ind_e,{250},:,:)) 
delete([/time, ind_s, ind_e, f3/])

;;;>-----------------------------------------------------------------------------------------<
;;;                                  季节平均 距平
;;;>-----------------------------------------------------------------------------------------<

n_year = dims_uvz(0)/12

hgt_200_anom = to_season(hgt_200)
u_200_anom = to_season(u_200)
v_200_anom = to_season(v_200)

;;;>-----------------------------------------------------------------------------------------<
;;;                                      回归
;;;>-----------------------------------------------------------------------------------------<

hgt_200_r = regCoef(pc, hgt_200_anom({latitude|:},{longitude|:},{time|:}))  
u_200_r = regCoef(pc, u_200_anom({latitude|:},{longitude|:},{time|:}))  
v_200_r = regCoef(pc, v_200_anom({latitude|:},{longitude|:},{time|:}))  

copy_VarCoords(hgt_200_anom(0,:,:), hgt_200_r)
copy_VarCoords(hgt_200_anom(0,:,:), u_200_r)
copy_VarCoords(hgt_200_anom(0,:,:), v_200_r)

;>---------------------------------------------------------------------------------------<
;;                    t-test
;>---------------------------------------------------------------------------------------<

p_h200 = student_t(hgt_200_r@tval, hgt_200_r@nptxy-2)
confi_rc_h200 = onedtond( tofloat(p_h200), (/nlat,nlon/))
copy_VarCoords(hgt_200_anom(0,:,:), confi_rc_h200)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

flbm = addfile("./LBM_Response_ASST.nc", "r")
u_lbm = flbm->u
v_lbm = flbm->v
hgt_lbm = flbm->hgt

;;;>-----------------------------------------------------------------------------------------<
;;;                                      绘图
;;;>-----------------------------------------------------------------------------------------<

res = True
res@gsnFrame = False
res@gsnDraw = False
res@gsnLeftString = ""
res@gsnRightString = ""
res@gsnCenterString = ""
res@gsnAddCyclic = True
res@tmBorderThicknessF = 2

res_sig = res
res_uv = res

res@vpWidthF = 0.7
res@vpHeightF = 0.35
res@mpShapeMode = "FreeAspect"
res@mpMinLatF = 20
res@mpMaxLatF = 90
res@mpMinLonF = 0
res@mpMaxLonF = 230
res@mpCenterLonF = 115
res@mpFillOn = False
res@mpGeophysicalLineColor = "gray40"
res@mpGeophysicalLineThicknessF = 2.0
res@gsnLeftStringOrthogonalPosF = 0.01
res@gsnLeftStringFontHeightF = 30
res@gsnRightStringOrthogonalPosF = 0.01
res@gsnRightStringFontHeightF = 30

res@tmXBMinorLengthF = 0
res@tmYLMinorLengthF = 0
res@tmXBMajorLengthF = 0.005
res@tmYLMajorLengthF = 0.005
res@tmXBLabelDeltaF = -0.5
res@tmYLLabelDeltaF = -0.5
res@tmXBLabelFontThicknessF = 1.5
res@tmYLLabelFontThicknessF = 1.5
res@tmXBLabelFontHeightF = 0.025
res@tmYLLabelFontHeightF = 0.025
res@tmXBTickSpacingF = 45
res@tmYLTickSpacingF = 20
res@tmXBMajorThicknessF = 2
res@tmYLMajorThicknessF = 2
res@tmXTOn = False
res@tmYROn = False

res@cnLinesOn = False
res@cnLineLabelsOn = False
res@cnInfoLabelOn = False
res@cnFillOn = True
res@cnLevelSelectionMode = "ExplicitLevels"
res@cnFillPalette = "MPL_BrBG"
res@cnFillColors = (/100,95,90,85,80,-1,-1,45,40,35,30,25/)

res@lbLabelBarOn = True
res@lbOrientation = "Vertical"
res@pmLabelBarWidthF = 0.06
res@lbLabelFontHeightF = 0.025
res@lbBoxEndCapStyle = "TriangleBothEnds"


res_uv@vcGlyphStyle = "CurlyVector"
res_uv@vcMinDistanceF = 0.02
res_uv@vcLineArrowThicknessF = 0.8

res_uv@vcRefAnnoOn = True
res_uv@vcRefLengthF = 0.045
res_uv@vcRefAnnoBackgroundColor = "white"
res_uv@vcRefAnnoPerimOn = True   ;关闭边框
res_uv@vcRefAnnoFontHeightF = 0.015
res_uv@vcRefAnnoString1On =   False ;设置标注箭头上方不显示注释
res_uv@vcRefAnnoString2On = True   ;设置标注箭头下方显示注释
res_uv@vcRefAnnoFontThicknessF = 1.5
res_uv@vcRefAnnoOrthogonalPosF = -0.19
res_uv@vcVectorDrawOrder = "PostDraw"
res_uv@vcLineArrowHeadMaxSizeF = 0.01
res_uv@vcLineArrowHeadMinSizeF = 0.01


;;  显著性检验场属性
res_sig@cnFillOn = True
res_sig@lbLabelBarOn = False
res_sig@cnLinesOn = False
res_sig@cnInfoLabelOn = False
res_sig@cnLineLabelsOn = False
res_sig@cnMonoFillPattern = False

res_sig@cnLevelSelectionMode = "ExplicitLevels"
res_sig@cnLevels = (/0.1/)
res_sig@cnFillPatterns = (/17, -1/)
res_sig@cnFillColors = (/"white", "Transparent"/)
res_sig@cnFillDotSizeF = 0.003
res_sig@cnFillDrawOrder = "PostDraw"


;;;>-----------------------------------------------------------------------------------------<

gres = True
gres@gsLineColor = "green"
gres@gsLineThicknessF = 3

gres2 = True
gres2@gsLineColor = "red"
gres2@gsLineThicknessF = 3

lonw1 = 80
lone1 = 170
lats1 = 35
latn1 = 42
x1 = (/lonw1,lone1,lone1,lonw1,lonw1/)
y1 = (/lats1,lats1,latn1,latn1,lats1/)

lonw13 = 30
lone13 = 170
lats13 = 70
latn13 = 83
x13 = (/lonw13,lone13,lone13,lonw13,lonw13/)
y13 = (/lats13,lats13,latn13,latn13,lats13/)

;;;>-----------------------------------------------------------------------------------------<

wks = gsn_open_wks("pdf", "./figs9")
    
shp = "/mnt/d/data/1-initial/altitude/青藏高原边界数据总集/TPBoundary_2500m/TPBoundary_2500m.shp"
tpres = True
tpres@gsFillColor = "grey45"
tpres@tfPolyDrawOrder = "PostDraw"

;;;>-----------------------------------------------------------------------------------------<

res@gsnLeftString = "(a) Reg_UVZ250 onto Russian Arctic SST"
res@cnLevels := (/-15,-12,-9,-6,-3,0,3,6,9,12,15/)
plot_h200 = gsn_csm_contour_map(wks, hgt_200_r, res)
plot_h200 = ColorNegDashZeroPosContour(plot_h200, "blue", "black", "red")
plot_sig = gsn_csm_contour(wks, confi_rc_h200, res_sig)
overlay(plot_h200, plot_sig)

res_uv@vcRefMagnitudeF := 1.0
res_uv@vcRefAnnoString2 := "1.0 m/s"
min200 = 0.2 
u_200_r = where(sqrt((/u_200_r/)^2 + (/v_200_r/)^2) .gt. min200, u_200_r, 0)
v_200_r = where(sqrt((/u_200_r/)^2 + (/v_200_r/)^2) .gt. min200, v_200_r, 0)
plot_uv200 = gsn_csm_vector(wks, u_200_r, v_200_r, res_uv)
overlay(plot_h200, plot_uv200)

dumz0 = gsn_add_polyline(wks, plot_uv200, x1, y1, gres)
dumz1 = gsn_add_polyline(wks, plot_uv200, x13, y13, gres2)

;;;>-----------------------------------------------------------------------------------------<

res@gsnLeftString = "(b) Resp_UVZ250 to Russian Arctic SST"
res@cnLevels := (/-2,-1.5,-1,-0.5,-0.1,0,0.1,0.5,1,1.5,2/)
plot_h200_lbm = gsn_csm_contour_map(wks, hgt_lbm, res)
plot_h200_lbm = ColorNegDashZeroPosContour(plot_h200_lbm, "blue", "black", "red")

res_uv@vcRefMagnitudeF := 0.03
res_uv@vcRefAnnoString2 := "0.03 m/s"
min200 = 0.002 
u_lbm = where(sqrt((/u_lbm/)^2 + (/v_lbm/)^2) .gt. min200, u_lbm, 0)
v_lbm = where(sqrt((/u_lbm/)^2 + (/v_lbm/)^2) .gt. min200, v_lbm, 0)
plot_uv200_lbm = gsn_csm_vector(wks, u_lbm, v_lbm, res_uv)
overlay(plot_h200_lbm, plot_uv200_lbm)

dumz0_lbm = gsn_add_polyline(wks, plot_uv200_lbm, x1, y1, gres)
dumz1_lbm = gsn_add_polyline(wks, plot_uv200_lbm, x13, y13, gres2)

;;;>-----------------------------------------------------------------------------------------<

gsn_panel(wks, (/plot_h200, plot_h200_lbm/), (/1, 2/), False)

;;;>-----------------------------------------------------------------------------------------<

end
