load "$NCARG_ROOT/lib/ncarg/nclscripts/contrib/run_cor.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

undef("comp_sig")
function comp_sig(var1, var2)  
local dims1, dims2, numid1, numid2, var1_mean, var2_mean, var1_std, var2_std, s1s2, df, t_m, confi_m

begin
    
    dims1 = dimsizes(var1)
    dims2 = dimsizes(var2)
    numid1 = dims1(0)
    numid2 = dims2(0)
    var1_mean = dim_avg_n_Wrap(var1, 0)
    var2_mean = dim_avg_n_Wrap(var2, 0)
    var1_std = dim_variance_n_Wrap(var1, 0)
    var2_std = dim_variance_n_Wrap(var2, 0)
    var1_std = where(var1_std.eq.0, var1_std@_FillValue, var1_std)
    var2_std = where(var2_std.eq.0, var2_std@_FillValue, var2_std)

    s1s2 = (numid1 - 1)*var1_std + (numid2 - 1)*var2_std
    df = numid1 + numid2 - 2
    t_m = (var2_mean - var1_mean)/(sqrt(s1s2/df) * sqrt(1.0/numid1 + 1.0/numid2))
    confi_m = var1_mean
    confi_m = student_t(t_m, df)

    return confi_m

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
    var_season = dtrend_msg_n(var_season&time, var_season, False, False, 0)

    return var_season
end

;;;>-----------------------------------------------------------------------------------------<
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

yyyymm_s = 197901
yyyymm_e = 202112

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

f2 = addfile("./HadISST_sst.nc", "r")    ;;  海温场
time = cd_calendar(f2->time, 1) 
ind_s = ind(time .eq. yyyymm_s)
ind_e = ind(time .eq. yyyymm_e)
sst = f2->sst(ind_s:ind_e,:,:)
sst = where(sst .eq. -1000.0, sst@_FillValue, sst)
delete([/time, ind_s, ind_e, f2/])

f4 = addfile("./ERA5_U_1x1_1940_2023_mon.nc", "r")    ;;  风场
time = cd_calendar(f4->time, 1) 
ind_s = ind(time .eq. yyyymm_s)
ind_e = ind(time .eq. yyyymm_e)
u200 = short2flt(f4->u(ind_s:ind_e,{250},:,:)) 
lat = u200&latitude
lon = u200&longitude
delete([/time, ind_s, ind_e, f4/])

;;;>-----------------------------------------------------------------------------------------<

var_season = to_season(sst)

var_season1 = var_season(0:12 ,:,:)   ;;;   p1
var_season2 = var_season(13:25,:,:)   ;;;   p2
var_season3 = var_season(26:,:,:)   ;;;   p3


sst_season = var_season
delete(var_season)

var_id1 = dim_avg_n_Wrap(var_season1, 0)
var_id2 = dim_avg_n_Wrap(var_season2, 0)
var_id3 = dim_avg_n_Wrap(var_season3, 0)
confi_sst_m1 = comp_sig(var_season1, var_season2)
confi_sst_m2 = comp_sig(var_season2, var_season3)
delete(var_season1)
delete(var_season2)
delete(var_season3)

sst_diff1 = var_id2 - var_id1
sst_diff2 = var_id2 - var_id3
sst3 = (sst_diff1 + sst_diff2)/2.
copy_VarCoords(var_id1, sst_diff1)
copy_VarCoords(var_id1, sst_diff2)
copy_VarCoords(var_id1, sst3)
delete(var_id1)
delete(var_id2)
delete(var_id3)

;;;>-----------------------------------------------------------------------------------------<

var_season = to_season(u200)

var_season1 = var_season(0:12 ,:,:)   ;;;   p1
var_season2 = var_season(13:25,:,:)   ;;;   p2
var_season3 = var_season(26:,:,:)   ;;;   p3

u200_season = var_season
delete(var_season)

var_id1 = dim_avg_n_Wrap(var_season1, 0)
var_id2 = dim_avg_n_Wrap(var_season2, 0)
var_id3 = dim_avg_n_Wrap(var_season3, 0)
confi_u200_m1 = comp_sig(var_season1, var_season2)
confi_u200_m2 = comp_sig(var_season2, var_season3)
delete(var_season1)
delete(var_season2)
delete(var_season3)

u200_diff1 = var_id2 - var_id1
u200_diff2 = var_id2 - var_id3
copy_VarCoords(var_id1, u200_diff1)
copy_VarCoords(var_id1, u200_diff2)
delete(var_id1)
delete(var_id2)
delete(var_id3)

u200_clim = to_clim(u200)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

fpc = addfile("./u250_index_ea_170.nc", "r")
pc = fpc->pc
pc = pc * (-1.)

f5 = addfile("./ERA5_PV_1x1_1940_2023_mon.nc", "r")  
time = cd_calendar(f5->time, 1) 
ind_s = ind(time .eq. yyyymm_s)
ind_e = ind(time .eq. yyyymm_e)
pv_250 = short2flt(f5->pv(ind_s:ind_e,{250},:,:))   ;;;  K m2 kg-1 s-1
lat = pv_250&latitude
delete([/time, ind_s, ind_e, f5/])

pv_season_250 = to_season(pv_250)
dims = dimsizes(pv_season_250)

pv_clm = to_clim(pv_250)

delete(pv_250)


R = 6371110 
rad = 0.0174533
dlat = (lat(1)-lat(0))*rad
dy = R*dlat

dpvdy = center_finite_diff_n(pv_season_250, dy, False, 0, 1)
copy_VarCoords(pv_season_250, dpvdy)
dpvdy_clm = center_finite_diff_n(pv_clm, dy, False, 0, 0)
copy_VarCoords(pv_clm, dpvdy_clm)

pv_season_250 = pv_season_250 * 10^7
pv_clm = pv_clm * 10^7
dpvdy = dpvdy * 10^12
dpvdy_clm = dpvdy_clm * 10^12


dpvdy_r = regCoef(pc, dpvdy({latitude|:},{longitude|:},{time|:}))  
copy_VarCoords(pv_season_250(0,:,:), dpvdy_r)

p_dpvdy = student_t(dpvdy_r@tval, dpvdy_r@nptxy-2)
confi_rc_dpvdy = onedtond( tofloat(p_dpvdy), (/dims(1),dims(2)/))
copy_VarCoords(pv_season_250(0,:,:), confi_rc_dpvdy)
confi_rc_dpvdy = where(confi_rc_dpvdy .eq. 0.0, confi_rc_dpvdy@_FillValue, confi_rc_dpvdy)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

res = True
res@gsnFrame = False
res@gsnDraw = False
res@gsnLeftString = ""
res@gsnRightString = ""
res@gsnCenterString = ""
res@gsnAddCyclic = True
res@tmBorderThicknessF = 2


res_sst = res
res_sig = res
res_sst = res_sst
res_clm = res

;;  降水场属性
res_sst@vpWidthF = 0.8
res_sst@vpHeightF = 0.35
res_sst@mpShapeMode = "FreeAspect"
res_sst@mpMinLatF = 20
res_sst@mpMaxLatF = 90
res_sst@mpCenterLonF = 180
res_sst@mpFillOn = False
res_sst@mpGeophysicalLineColor = "gray40"
res_sst@mpGeophysicalLineThicknessF = 1.0
res_sst@gsnLeftStringOrthogonalPosF = 0.01
res_sst@gsnLeftStringFontHeightF = 33
res_sst@gsnRightStringOrthogonalPosF = 0.01
res_sst@gsnRightStringFontHeightF = 33

res_sst@tmXBMinorLengthF = 0
res_sst@tmYLMinorLengthF = 0
res_sst@tmXBMajorLengthF = 0.005
res_sst@tmYLMajorLengthF = 0.005
res_sst@tmXBLabelDeltaF = -0.5
res_sst@tmYLLabelDeltaF = -0.5
res_sst@tmXBLabelFontThicknessF = 1.5
res_sst@tmYLLabelFontThicknessF = 1.5
res_sst@tmXBLabelFontHeightF = 0.027
res_sst@tmYLLabelFontHeightF = 0.027
res_sst@tmXBTickSpacingF = 45
res_sst@tmYLTickSpacingF = 20
res_sst@tmXBMajorThicknessF = 1.5
res_sst@tmYLMajorThicknessF = 1.5
res_sst@tmXTOn = False
res_sst@tmYROn = False
resu = res_sst 

res_sst@cnLinesOn = False
res_sst@cnLineLabelsOn = False
res_sst@cnInfoLabelOn = False
res_sst@cnFillOn = True

res_sst@cnFillPalette = "BlWhRe"
res_sst@cnLevelSelectionMode = "ExplicitLevels"
res_sst@cnLevels = (/-1.0,-0.7,-0.5,-0.3,-0.1,0,0.1,0.3,0.5,0.7,1.0/)   ;;; raw
res_sst@cnFillColors = (/33,35,37,39,41,-1,-1,56,59,62,65,68/)
res_sst@cnFillDrawOrder = "PreDraw"
res_sst@lbLabelFontHeightF = 0.025
res_sst@lbLabelBarOn = True
res_sst@lbOrientation = "Vertical"
res_sst@pmLabelBarWidthF = 0.06
res_sst@lbBoxEndCapStyle = "TriangleBothEnds"

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


resu@cnFillOn = True
resu@cnLineLabelsOn = False
resu@cnLinesOn = False
resu@cnFillPalette = "BlWhRe"
resu@cnFillOpacityF = 0.8
resu@cnLevelSelectionMode = "ExplicitLevels"
resu@cnLevels = (/-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5/)
resu@cnFillColors = (/33,35,37,39,41,-1,-1,56,59,62,65,68/)

resu@lbLabelFontHeightF = 0.025
resu@lbLabelBarOn = True
resu@lbOrientation = "Vertical"
resu@pmLabelBarWidthF = 0.06
resu@lbBoxEndCapStyle = "TriangleBothEnds"

res_clm@cnLinesOn = True
res_clm@cnLineLabelsOn = True
res_clm@cnInfoLabelOn = False
res_clm@cnFillOn = False
res_clm@cnLineLabelFontHeightF = 0.008
res_clm@cnLineLabelBackgroundColor = -1
res_clm@cnLineLabelAngleF = 0.0
res_clm@cnLineLabelDensityF = 1
res_clm@cnLineLabelInterval = 1
res_clm@cnLineLabelPerimOn = False
res_clm@cnLineLabelFormat = "0@;*.2f"
res_clm@cnLineDrawOrder = "PostDraw"
res_clm@cnLabelDrawOrder = "PostDraw"
res_clm@cnLevelSelectionMode = "ExplicitLevels"
res_clm@cnLevels = (/20,25,30/)
res_clm@cnLineThicknessF = 2

;;;>-----------------------------------------------------------------------------------------<

gres = True
gres@gsLineThicknessF = 2


lonw1 = 80
lone1 = 170
lats1 = 35
latn1 = 42
x1 = (/lonw1,lone1,lone1,lonw1,lonw1/)
y1 = (/lats1,lats1,latn1,latn1,lats1/)
u200_ea = dim_avg_n_Wrap(u200_season(:,{lats1:latn1},{lonw1:lone1}), (/1, 2/))
u200_ea = dim_standardize(u200_ea, 1)

; ff = addfile("./u250_index_ea_170.nc", "c")
; ff->pc = u200_ea

;;;   PV north Pacific
lonw2 = 100
lone2 = 220
lats2 = 45
latn2 = 65
x2 = (/lonw2,lone2,lone2,lonw2,lonw2/)
y2 = (/lats2,lats2,latn2,latn2,lats2/)

;;;>-----------------------------------------------------------------------------------------<

lonw13 = 30
lone13 = 170
lats13 = 70
latn13 = 83
x13 = (/lonw13,lone13,lone13,lonw13,lonw13/)
y13 = (/lats13,lats13,latn13,latn13,lats13/)
sst_polar = dim_avg_n_Wrap(sst_season(:,{lats13:latn13},{lonw13:lone13}), (/1, 2/))
sst_polar = dim_standardize(sst_polar, 1)

; ff2 = addfile("./sst_index_polar.nc", "c")
; ff2->pc = sst_polar

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

wks = gsn_open_wks("pdf", "fig3")
    
pmres = True
pmres@gsMarkerIndex = 12
pmres@gsMarkerSizeF = 0.02
pmres@gsMarkerThicknessF = 3.5
pmres@gsMarkerColor = "green"

;;;>-----------------------------------------------------------------------------------------<

gres@gsLineColor = "green"
resu@gsnLeftString = "(a) U250 P2 minus P1"
plotu1 = gsn_csm_contour_map(wks, u200_diff1, resu)
plotu_sig1 = gsn_csm_contour(wks, confi_u200_m1, res_sig)
plotu_clim1 = gsn_csm_contour(wks, u200_clim, res_clm)
overlay(plotu1, plotu_sig1)
overlay(plotu1, plotu_clim1)
dum1 = gsn_add_polyline(wks, plotu_clim1, x1, y1, gres)

;;;>-----------------------------------------------------------------------------------------<

resu@gsnLeftString = "(b) U250 P2 minus P3"
plotu2 = gsn_csm_contour_map(wks, u200_diff2, resu)
plotu_sig2 = gsn_csm_contour(wks, confi_u200_m2, res_sig)
plotu_clim2 = gsn_csm_contour(wks, u200_clim, res_clm)
overlay(plotu2, plotu_sig2)
overlay(plotu2, plotu_clim2)
dum2 = gsn_add_polyline(wks, plotu_clim2, x1, y1, gres)

;;;>-----------------------------------------------------------------------------------------<

gres@gsLineColor = "red"
res_sst@gsnLeftString = "(e) SST P2 minus P1"
plotsst1 = gsn_csm_contour_map(wks, sst_diff1, res_sst)
plotsst_sig1 = gsn_csm_contour(wks, confi_sst_m1, res_sig)
overlay(plotsst1, plotsst_sig1)
dum3 = gsn_add_polyline(wks, plotsst_sig1, x13, y13, gres)

;;;>-----------------------------------------------------------------------------------------<

res_sst@gsnLeftString = "(f) SST P2 minus P3"
plotsst2 = gsn_csm_contour_map(wks, sst_diff2, res_sst)
plotsst_sig2 = gsn_csm_contour(wks, confi_sst_m2, res_sig)
overlay(plotsst2, plotsst_sig2)
dum132 = gsn_add_polyline(wks, plotsst_sig2, x13, y13, gres)

;;;>-----------------------------------------------------------------------------------------<

resu@gsnLeftString = "(d) Clim_dPVdy250 "
resu@cnFillPalette = "MPL_BuGn"
resu@cnLevels := (/1,1.5,2,2.5,3/)
resu@cnFillColors := (/-1,40,60,80,100,120/)
resu@cnFillOpacityF = 1.0
plot_dpvdy_clm = gsn_csm_contour_map(wks, dpvdy_clm, resu)

;;;>-----------------------------------------------------------------------------------------<

gres@gsLineColor = "purple"
resu@gsnLeftString = "(c) Reg_dPVdy250 onto EAJ"
resu@cnFillPalette = "MPL_BrBG"
resu@cnLevels := (/-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5/)
resu@cnFillColors := (/95,90,85,80,75,-1,-1,45,40,35,30,25/)
plot_dpvdy_reg = gsn_csm_contour_map(wks, dpvdy_r, resu)
plot_dpvdy_sig = gsn_csm_contour(wks, confi_rc_dpvdy, res_sig)
overlay(plot_dpvdy_reg, plot_dpvdy_sig)
dum11 = gsn_add_polyline(wks, plot_dpvdy_sig, x2, y2, gres)

;;;>-----------------------------------------------------------------------------------------<

pres = True
pres@gsnPanelYWhiteSpacePercent = 5
pres@gsnPanelXWhiteSpacePercent = 3
gsn_panel(wks, (/plotu1, plotu2, plot_dpvdy_reg, plot_dpvdy_clm, plotsst1, plotsst2/), (/3, 2/), pres)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

end

