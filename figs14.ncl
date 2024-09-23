load "$NCARG_ROOT/lib/ncarg/nclscripts/contrib/run_cor.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

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
    var1_std = where(var1_std.eq.0., var1_std@_FillValue, var1_std)
    var2_std = where(var2_std.eq.0., var2_std@_FillValue, var2_std)

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
;;;>-----------------------------------------------------------------------------------------<

function clim_diff(var)     ;;;   
local yyyymm_s_p1, yyyymm_e_p1, yyyymm_s_p2, yyyymm_e_p2, yyyymm_s_p3, yyyymm_e_p3, var_season, \
        var_season_id1, var_season_id2, var_season_id3, confi_var_m21, confi_var_m23, var_id1, \
        var_id2, var_id3, var_diff21, var_diff23
begin

    yyyymm_s_p1 = 197901
    yyyymm_e_p1 = 199112

    yyyymm_s_p2 = 199201
    yyyymm_e_p2 = 200412

    yyyymm_s_p3 = 200501
    yyyymm_e_p3 = 202112

    var_season = to_season(var)

    var_season_id1 = var_season(yyyymm_s_p1/100-1979:yyyymm_e_p1/100-1979,:,:)
    var_season_id2 = var_season(yyyymm_s_p2/100-1979:yyyymm_e_p2/100-1979,:,:)
    var_season_id3 = var_season(yyyymm_s_p3/100-1979:yyyymm_e_p3/100-1979,:,:)
    delete(var_season)

    confi_var_m21 = comp_sig(var_season_id2, var_season_id1)
    confi_var_m23 = comp_sig(var_season_id2, var_season_id3)

    var_id1 = dim_avg_n_Wrap(var_season_id1, 0)
    var_id2 = dim_avg_n_Wrap(var_season_id2, 0)
    var_id3 = dim_avg_n_Wrap(var_season_id3, 0)

    var_diff21 = var_id2 - var_id1
    var_diff23 = var_id2 - var_id3
    copy_VarCoords(var_id1, var_diff21)
    copy_VarCoords(var_id1, var_diff23)
    delete([/var_season_id1, var_season_id2, var_season_id3, var_id1, var_id2, var_id3/])

    return [/var_diff21, confi_var_m21, var_diff23, confi_var_m23/]
end

;;;>-----------------------------------------------------------------------------------------<

begin

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

yyyymm_s = 197901
yyyymm_e = 202112


f2 = addfile("./HadISST_sst.nc", "r")    ;;  海温场
time = cd_calendar(f2->time, 1) 
ind_s = ind(time .eq. yyyymm_s)
ind_e = ind(time .eq. yyyymm_e)
sst = f2->sst(ind_s:ind_e,:,:)
sst = where(sst .eq. -1000.0, sst@_FillValue, sst)
delete([/time, ind_s, ind_e, f2/])


f5 = addfile("./olr.mon.mean.nc", "r")
time = cd_calendar(f5->time, 1) 
ind_s = ind(time .eq. yyyymm_s)
ind_e = ind(time .eq. yyyymm_e)
olr = f5->olr(ind_s:ind_e,:,:)
delete([/time, ind_s, ind_e, f5/])

f5 = addfile("./gpcp_v02r03_monthly_197901_202112.nc", "r")
time = cd_calendar(f5->time, 1) 
ind_s = ind(time .eq. yyyymm_s)
ind_e = ind(time .eq. yyyymm_e)
pre = f5->precip(ind_s:ind_e,:,:)
delete([/time, ind_s, ind_e, f5/])

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<


;;;>-----------------------------------------------------------------------------------------<

temp = clim_diff(sst)
diff_sst_m21 = temp[0]
confi_sst_m21 = temp[1]
diff_sst_m23 = temp[2]
confi_sst_m23 = temp[3]
delete(temp)

temp = clim_diff(olr)
diff_olr_m21 = temp[0]
confi_olr_m21 = temp[1]
diff_olr_m23 = temp[2]
confi_olr_m23 = temp[3]
delete(temp)

temp = clim_diff(pre)
diff_pre_m21 = temp[0]
confi_pre_m21 = temp[1]
diff_pre_m23 = temp[2]
confi_pre_m23 = temp[3]
delete(temp)

;;;>-----------------------------------------------------------------------------------------<

olr_season = to_season(olr)

gres = True
gres@gsLineColor = "blue"
gres@gsLineThicknessF = 2

lonw1 = 45
lone1 = 90
lats1 = -10
latn1 = 10
x2 = (/lonw1,lone1,lone1,lonw1,lonw1/)
y2 = (/lats1,lats1,latn1,latn1,lats1/)

olr_tropical_index = dim_avg_n_Wrap(olr_season(:,{lats1:latn1},{lonw1:lone1}), (/1, 2/))
olr_tropical_index = dim_standardize(olr_tropical_index, 1)

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
res_clm = res

;;  降水场属性
res_sst@vpWidthF = 0.8
res_sst@vpHeightF = 0.3
res_sst@mpShapeMode = "FreeAspect"
res_sst@mpMinLatF = -40
res_sst@mpMaxLatF = 40
res_sst@mpCenterLonF = 180
res_sst@mpFillOn = False
res_sst@mpGeophysicalLineColor = "gray40"
res_sst@mpGeophysicalLineThicknessF = 1.5
res_sst@gsnLeftStringOrthogonalPosF = 0.01
res_sst@gsnLeftStringFontHeightF = 25
res_sst@gsnRightStringOrthogonalPosF = 0.01
res_sst@gsnRightStringFontHeightF = 25

res_sst@tmXBMinorLengthF = 0
res_sst@tmYLMinorLengthF = 0
res_sst@tmXBMajorLengthF = 0.005
res_sst@tmYLMajorLengthF = 0.005
res_sst@tmXBLabelDeltaF = -0.5
res_sst@tmYLLabelDeltaF = -0.5
res_sst@tmXBLabelFontThicknessF = 1.5
res_sst@tmYLLabelFontThicknessF = 1.5
res_sst@tmXBLabelFontHeightF = 0.02
res_sst@tmYLLabelFontHeightF = 0.02
res_sst@tmXBTickSpacingF = 45
res_sst@tmYLTickSpacingF = 20
res_sst@tmXBMajorThicknessF = 2.0
res_sst@tmYLMajorThicknessF = 2.0
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
res_sst@lbLabelFontHeightF = 0.02
res_sst@lbLabelBarOn = True
res_sst@lbOrientation = "Vertical"
res_sst@pmLabelBarWidthF = 0.04
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
resu@cnFillPalette = "testcmap"
resu@cnLevelSelectionMode = "ExplicitLevels"

resu@lbLabelFontHeightF = 0.02
resu@lbLabelBarOn = True
resu@lbOrientation = "Vertical"
resu@pmLabelBarWidthF = 0.04
resu@lbBoxEndCapStyle = "TriangleBothEnds"

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

wks = gsn_open_wks("pdf", "figs14")
    

res_sst@gsnLeftString = "(a) Diff_SST (P2 minus P1)"
plot_sst_21 = gsn_csm_contour_map(wks, diff_sst_m21, res_sst)
plot_sst_21_sig = gsn_csm_contour(wks, confi_sst_m21, res_sig)
overlay(plot_sst_21, plot_sst_21_sig)

res_sst@gsnLeftString = "(b) Diff_SST (P2 minus P3)"
plot_sst_23 = gsn_csm_contour_map(wks, diff_sst_m23, res_sst)
plot_sst_23_sig = gsn_csm_contour(wks, confi_sst_m23, res_sig)
overlay(plot_sst_23, plot_sst_23_sig)

;;;>-----------------------------------------------------------------------------------------<

colors = (/155, 145, 135, 125, 115, 105, -1,-1, 94, 90, 84, 78, 72, 66/)
resu@cnLevels = (/-6, -5.0, -4.0, -3.0, -2.0, -1.0,0, 1.0, 2.0, 3.0, 4.0, 5.0, 6/)
resu@cnFillColors = colors(::-1)
resu@gsnLeftString = "(c) Diff_OLR (P2 minus P1)"
plot_olr_21 = gsn_csm_contour_map(wks, diff_olr_m21, resu)
plot_olr_21_sig = gsn_csm_contour(wks, confi_olr_m21, res_sig)
overlay(plot_olr_21, plot_olr_21_sig)
dum1 = gsn_add_polyline(wks, plot_olr_21_sig, x2, y2, gres)

resu@gsnLeftString = "(d) Diff_OLR (P2 minus P3)"
plot_olr_23 = gsn_csm_contour_map(wks, diff_olr_m23, resu)
plot_olr_23_sig = gsn_csm_contour(wks, confi_olr_m23, res_sig)
overlay(plot_olr_23, plot_olr_23_sig)
dum2 = gsn_add_polyline(wks, plot_olr_23_sig, x2, y2, gres)

;;;>-----------------------------------------------------------------------------------------<

resu@cnLevels := (/-1.0, -0.8, -0.6, -0.4, -0.2, -0.1,0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0/)
resu@cnFillColors = colors
resu@gsnLeftString = "(e) Diff_Pr (P2 minus P1)"
plot_pre_21 = gsn_csm_contour_map(wks, diff_pre_m21, resu)
plot_pre_21_sig = gsn_csm_contour(wks, confi_pre_m21, res_sig)
overlay(plot_pre_21, plot_pre_21_sig)
dum3 = gsn_add_polyline(wks, plot_pre_21_sig, x2, y2, gres)

resu@gsnLeftString = "(f) Diff_Pr (P2 minus P3)"
plot_pre_23 = gsn_csm_contour_map(wks, diff_pre_m23, resu)
plot_pre_23_sig = gsn_csm_contour(wks, confi_pre_m23, res_sig)
overlay(plot_pre_23, plot_pre_23_sig)
dum4 = gsn_add_polyline(wks, plot_pre_23_sig, x2, y2, gres)


;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

pres = True
pres@gsnPanelYWhiteSpacePercent = 5
gsn_panel(wks, (/plot_sst_21, plot_sst_23, plot_olr_21, plot_olr_23, plot_pre_21, plot_pre_23/), (/3, 2/), pres)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

end

