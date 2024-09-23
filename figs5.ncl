procedure add_lc_labels(wks,map,minlat,maxlat,minlon,maxlon,\
                        lat_spacing[1]:integer,lon_spacing[1]:integer)
    local lat_values, nlat, lat1_ndc, lat2_ndc, lon1_ndc, lon2_ndc,slope,txres, \
    lon_values, PI, RAD_TO_DEG, dum_lft, dum_rgt, dum_bot
    begin
      PI         = 3.14159
      RAD_TO_DEG = 180./PI

    ;---Pick some "nice" values for the latitude labels.
      lat_values = ispan(toint(minlat),toint(maxlat),lat_spacing) * 1.
      nlat       = dimsizes(lat_values)
    ;
    ; We need to get the slope of the left and right min/max longitude lines.
    ; Use NDC coordinates to do this.
    ;
      lat1_ndc = new(1,float)
      lon1_ndc = new(1,float)
      lat2_ndc = new(1,float)
      lon2_ndc = new(1,float)
      datatondc(map,minlon,lat_values(0),lon1_ndc,lat1_ndc)
      datatondc(map,minlon,lat_values(nlat-1),lon2_ndc,lat2_ndc)
      if(lon2_ndc.eq.lon1_ndc) then
          print("add_lc_labels: error: not enough valid latitude values to generate labels")
          print("    You may need to decrease the lat_spacing")
          exit
      end if
      slope_lft = (lat2_ndc-lat1_ndc)/(lon2_ndc-lon1_ndc)

      datatondc(map,maxlon,lat_values(0),lon1_ndc,lat1_ndc)
      datatondc(map,maxlon,lat_values(nlat-1),lon2_ndc,lat2_ndc)
      if(lat2_ndc.eq.lat1_ndc) then
          print("add_lc_labels: error: not enough valid longitude values to generate labels")
          print("    You may need to decrease the lon_spacing")
          exit
      end if
      slope_rgt = (lat2_ndc-lat1_ndc)/(lon2_ndc-lon1_ndc)

    ;---Set some text resources
      txres               = True
      txres@txFontHeightF = 0.027
      txres@txPosXF       = 0.1
    ;
    ; Loop through lat values, and attach labels to the left and
    ; right edges of the masked LC plot.  The labels will be 
    ; rotated to fit the line better.
    ;
      dum_lft = new(nlat,graphic)    ; Dummy array to hold attached strings.
      dum_rgt = new(nlat,graphic)    ; Dummy array to hold attached strings.
      do n=0,nlat-1
    ; Add extra white space to labels.
        lat_label_rgt = "          "  + lat_values(n); + "~S~o~N~"

    ;---Check if North, South, or Zero
        if(lat_values(n).lt.0) then
          ; lat_label_lft = lat_values(n) + "~S~o~N~S          "
          lat_label_lft = lat_values(n) + "S          "
          lat_label_rgt = lat_label_rgt + "S"
        end if
        if(lat_values(n).gt.0) then
          ; lat_label_lft = lat_values(n) + "~S~o~N~N          "
          lat_label_lft = lat_values(n) + "N          "
          lat_label_rgt = lat_label_rgt + "N"
        end if
        if(lat_values(n).eq.0) then
          ; lat_label_lft = lat_values(n); + "~S~o~N~           "
          lat_label_lft = lat_values(n)
        end if

    ;---Left label
        txres@txAngleF = RAD_TO_DEG * atan(slope_lft) - 90
        dum_lft(n) = gsn_add_text(wks,map,lat_label_lft,minlon,lat_values(n),txres)

    ;---Right label
        txres@txAngleF = RAD_TO_DEG * atan(slope_rgt) + 90
        dum_rgt(n) = gsn_add_text(wks,map,lat_label_rgt,maxlon,lat_values(n),txres)
      end do

    ;----------------------------------------------------------------------
    ; Now do longitude labels. These are harder because we're not
    ; adding them to a straight line.
    ;
    ; Loop through lon values, and attach labels to the bottom edge of the
    ; masked LC plot.
    ;
      delete(txres@txPosXF)
      txres@txPosYF = -5.0

    ;---Pick some "nice" values for the longitude labels.
      lon_values = ispan(toint(minlon+10),toint(maxlon-10),lon_spacing) * 1.
      nlon       = dimsizes(lon_values)

      dum_bot = new(nlon,graphic)    ; Dummy array to hold attached strings.

      do n=0,nlon-1
    ;
    ; For each longitude label, we need to figure out how much to rotate
    ; it, so get the approximate slope at that point.
    ;
        datatondc(map,lon_values(n)-0.25,minlat,lon1_ndc,lat1_ndc)
        datatondc(map,lon_values(n)+0.25,minlat,lon2_ndc,lat2_ndc)
        slope_bot = (lat1_ndc-lat2_ndc)/(lon1_ndc-lon2_ndc)
        txres@txAngleF  = atan(slope_bot) * RAD_TO_DEG
    ;
    ; Create longitude label. Add extra carriage returns to
    ; move label away from plot.
    ;
    ;---Check if East, West, or Zero
        ; lon_label_bot = "  ~C~  ~C~" + abs(lon_values(n)) + "~S~o~N~"
        lon_label_bot = "  ~C~  ~C~" + abs(lon_values(n))
        if(lon_values(n).lt.0) then
          lon_label_bot = lon_label_bot + "W"
        end if
        if(lon_values(n).gt.0) then
          lon_label_bot = lon_label_bot + "E"
        end if

    ;---Attach to map.
        dum_bot(n) = gsn_add_text(wks,map,lon_label_bot,lon_values(n),minlat,txres)
      end do
    end

;;;>-----------------------------------------------------------------------------------------<

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

; undef confi(var)
function confi(var)     ;;;   返回t检验结果
    local p, dims, confi_var
    begin

        dims = dimsizes(var)
        p = student_t(var@tval, var@nptxy-2)
        confi_var = onedtond( tofloat(p), (/dims(0),dims(1)/) )
        copy_VarCoords(var, confi_var)

        return confi_var
    end

;;;>-----------------------------------------------------------------------------------------<

begin

;;;>-----------------------------------------------------------------------------------------<
;;;                                时间段
;;;>-----------------------------------------------------------------------------------------<

yyyymm_s_p1 = 197901
yyyymm_e_p1 = 199112

yyyymm_s_p2 = 199201
yyyymm_e_p2 = 200412

yyyymm_s_p3 = 200501
yyyymm_e_p3 = 202112

;;;   是否对环流场进行去年代际信号处理
rm_id_circ = False

data = 1   ;;;   1 JRA55 ;  2 ERA5

;;;>-----------------------------------------------------------------------------------------<
;;;                               读取数据
;;;>-----------------------------------------------------------------------------------------<

f_pc = addfile("./olr_index.nc", "r")
pc_p1 = f_pc->pc(yyyymm_s_p1/100-1979:yyyymm_e_p1/100-1979)
pc_p2 = f_pc->pc(yyyymm_s_p2/100-1979:yyyymm_e_p2/100-1979)
pc_p3 = f_pc->pc(yyyymm_s_p3/100-1979:yyyymm_e_p3/100-1979)

pc_p1 = pc_p1 * (-1)
pc_p1 = dim_standardize(pc_p1, 1)

pc_p2 = pc_p2 * (-1)
pc_p2 = dim_standardize(pc_p2, 1)

pc_p3 = pc_p3 * (-1)
pc_p3 = dim_standardize(pc_p3, 1)


;;;>-----------------------------------------------------------------------------------------<

yyyymm_s = 197901
yyyymm_e = 202112

if (data .eq. 1) then

    idir = "./JRA55/mon/"

    f1 = addfile(idir + "pressure/anl_p125.007_hgt.195801_202312.nc", "r") 
    time = cd_calendar(f1->initial_time0_hours, 1) 
    ind_s = ind(time .eq. yyyymm_s)
    ind_e = ind(time .eq. yyyymm_e)
    hgt_200 = f1->HGT_GDS0_ISBL_S123(ind_s:ind_e,{250},:,:)
    hgt_850 = f1->HGT_GDS0_ISBL_S123(ind_s:ind_e,{850},:,:)

    dims_uvz = dimsizes(hgt_200)
    delete([/time, ind_s, ind_e, f1/])

    f2 = addfile(idir + "pressure/anl_p125.033_ugrd.195801_202312.nc", "r")
    time = cd_calendar(f2->initial_time0_hours, 1) 
    ind_s = ind(time .eq. yyyymm_s)
    ind_e = ind(time .eq. yyyymm_e)
    u_200 = f2->UGRD_GDS0_ISBL_S123(ind_s:ind_e,{250},:,:) 
    u_850 = f2->UGRD_GDS0_ISBL_S123(ind_s:ind_e,{850},:,:) 
    delete([/time, ind_s, ind_e, f2/])

    f3 = addfile(idir + "pressure/anl_p125.034_vgrd.195801_202312.nc", "r")  
    time = cd_calendar(f3->initial_time0_hours, 1) 
    ind_s = ind(time .eq. yyyymm_s)
    ind_e = ind(time .eq. yyyymm_e)
    v_200 = f3->VGRD_GDS0_ISBL_S123(ind_s:ind_e,{250},:,:) 
    v_850 = f3->VGRD_GDS0_ISBL_S123(ind_s:ind_e,{850},:,:) 
    delete([/time, ind_s, ind_e, f3/])

    ; f4 = addfile(idir + "surface/anl_surf125.195801_202311.nc", "r")
    f4 = addfile(idir + "surface/anl_surf125.011_tmp.195801_202312.nc", "r")
    time = cd_calendar(f4->initial_time0_hours, 1) 
    ind_s = ind(time .eq. yyyymm_s)
    ind_e = ind(time .eq. yyyymm_e)
    t2m = f4->TMP_GDS0_HTGL_S123(ind_s:ind_e,:,:)  ;;  K
    dims_t2m = dimsizes(t2m)
    delete([/time, ind_s, ind_e, f4/])

else

    idir = "./era5/monthly/"

    f1 = addfile(idir + "ERA5_Z_1x1_1940_2023_mon.nc", "r") 
    time = cd_calendar(f1->time, 1) 
    ind_s = ind(time .eq. yyyymm_s)
    ind_e = ind(time .eq. yyyymm_e)
    hgt_200 = short2flt(f1->z(ind_s:ind_e,{250},:,:))
    hgt_850 = short2flt(f1->z(ind_s:ind_e,{850},:,:))
    hgt_200 = hgt_200/9.8
    hgt_850 = hgt_850/9.8
    dims_uvz = dimsizes(hgt_200)
    delete([/time, ind_s, ind_e, f1/])

    f2 = addfile(idir + "ERA5_U_1x1_1940_2023_mon.nc", "r")
    time = cd_calendar(f2->time, 1) 
    ind_s = ind(time .eq. yyyymm_s)
    ind_e = ind(time .eq. yyyymm_e)
    u_200 = short2flt(f2->u(ind_s:ind_e,{250},:,:)) 
    u_850 = short2flt(f2->u(ind_s:ind_e,{850},:,:)) 
    delete([/time, ind_s, ind_e, f2/])

    f3 = addfile(idir + "ERA5_V_1x1_1940_2023_mon.nc", "r")  
    time = cd_calendar(f3->time, 1) 
    ind_s = ind(time .eq. yyyymm_s)
    ind_e = ind(time .eq. yyyymm_e)
    v_200 = short2flt(f3->v(ind_s:ind_e,{250},:,:)) 
    v_850 = short2flt(f3->v(ind_s:ind_e,{850},:,:)) 
    delete([/time, ind_s, ind_e, f3/])

    f4 = addfile(idir + "ERA5_2mTSLP_1x1_1940_2023_mon.nc", "r")
    time = cd_calendar(f4->time, 1) 
    ind_s = ind(time .eq. yyyymm_s)
    ind_e = ind(time .eq. yyyymm_e)
    t2m = short2flt(f4->t2m(ind_s:ind_e,:,:))
    dims_t2m = dimsizes(t2m)
    delete([/time, ind_s, ind_e, f4/])

end if


;;;>-----------------------------------------------------------------------------------------<
;;;                                  季节平均 距平
;;;>-----------------------------------------------------------------------------------------<

n_year = dims_uvz(0)/12

hgt_200_anom = to_season(hgt_200)
hgt_850_anom = to_season(hgt_850)
u_200_anom = to_season(u_200)
u_850_anom = to_season(u_850)
v_200_anom = to_season(v_200)
v_850_anom = to_season(v_850)
t2m_anom = to_season(t2m)

u_200_clim = to_clim(u_200)
v_200_clim = to_clim(v_200)

delete([/hgt_200, hgt_850, u_200, u_850, v_200, v_850, t2m/])

;;;>-----------------------------------------------------------------------------------------<
;;;                                      回归
;;;>-----------------------------------------------------------------------------------------<

hgt_200_r_p1 = regCoef_n(pc_p1, hgt_200_anom(yyyymm_s_p1/100-1979:yyyymm_e_p1/100-1979,:,:), 0, 0)    
hgt_200_r_p2 = regCoef_n(pc_p2, hgt_200_anom(yyyymm_s_p2/100-1979:yyyymm_e_p2/100-1979,:,:), 0, 0)    
hgt_200_r_p3 = regCoef_n(pc_p3, hgt_200_anom(yyyymm_s_p3/100-1979:yyyymm_e_p3/100-1979,:,:), 0, 0)    
hgt_850_r_p1 = regCoef_n(pc_p1, hgt_850_anom(yyyymm_s_p1/100-1979:yyyymm_e_p1/100-1979,:,:), 0, 0)    
hgt_850_r_p2 = regCoef_n(pc_p2, hgt_850_anom(yyyymm_s_p2/100-1979:yyyymm_e_p2/100-1979,:,:), 0, 0)    
hgt_850_r_p3 = regCoef_n(pc_p3, hgt_850_anom(yyyymm_s_p3/100-1979:yyyymm_e_p3/100-1979,:,:), 0, 0)   
u_200_r_p1 = regCoef_n(pc_p1, u_200_anom(yyyymm_s_p1/100-1979:yyyymm_e_p1/100-1979,:,:), 0, 0)    
u_200_r_p2 = regCoef_n(pc_p2, u_200_anom(yyyymm_s_p2/100-1979:yyyymm_e_p2/100-1979,:,:), 0, 0)    
u_200_r_p3 = regCoef_n(pc_p3, u_200_anom(yyyymm_s_p3/100-1979:yyyymm_e_p3/100-1979,:,:), 0, 0)   
u_850_r_p1 = regCoef_n(pc_p1, u_850_anom(yyyymm_s_p1/100-1979:yyyymm_e_p1/100-1979,:,:), 0, 0)    
u_850_r_p2 = regCoef_n(pc_p2, u_850_anom(yyyymm_s_p2/100-1979:yyyymm_e_p2/100-1979,:,:), 0, 0)    
u_850_r_p3 = regCoef_n(pc_p3, u_850_anom(yyyymm_s_p3/100-1979:yyyymm_e_p3/100-1979,:,:), 0, 0)   
v_200_r_p1 = regCoef_n(pc_p1, v_200_anom(yyyymm_s_p1/100-1979:yyyymm_e_p1/100-1979,:,:), 0, 0)    
v_200_r_p2 = regCoef_n(pc_p2, v_200_anom(yyyymm_s_p2/100-1979:yyyymm_e_p2/100-1979,:,:), 0, 0)    
v_200_r_p3 = regCoef_n(pc_p3, v_200_anom(yyyymm_s_p3/100-1979:yyyymm_e_p3/100-1979,:,:), 0, 0)   
v_850_r_p1 = regCoef_n(pc_p1, v_850_anom(yyyymm_s_p1/100-1979:yyyymm_e_p1/100-1979,:,:), 0, 0)    
v_850_r_p2 = regCoef_n(pc_p2, v_850_anom(yyyymm_s_p2/100-1979:yyyymm_e_p2/100-1979,:,:), 0, 0)    
v_850_r_p3 = regCoef_n(pc_p3, v_850_anom(yyyymm_s_p3/100-1979:yyyymm_e_p3/100-1979,:,:), 0, 0) 
t2m_r_p1 = regCoef_n(pc_p1, t2m_anom(yyyymm_s_p1/100-1979:yyyymm_e_p1/100-1979,:,:), 0, 0)    
t2m_r_p2 = regCoef_n(pc_p2, t2m_anom(yyyymm_s_p2/100-1979:yyyymm_e_p2/100-1979,:,:), 0, 0)    
t2m_r_p3 = regCoef_n(pc_p3, t2m_anom(yyyymm_s_p3/100-1979:yyyymm_e_p3/100-1979,:,:), 0, 0)   

copy_VarCoords(hgt_200_anom(0,:,:), hgt_200_r_p1)
copy_VarCoords(hgt_200_anom(0,:,:), hgt_200_r_p2)
copy_VarCoords(hgt_200_anom(0,:,:), hgt_200_r_p3)
copy_VarCoords(hgt_200_anom(0,:,:), hgt_850_r_p1)
copy_VarCoords(hgt_200_anom(0,:,:), hgt_850_r_p2)
copy_VarCoords(hgt_200_anom(0,:,:), hgt_850_r_p3)
copy_VarCoords(hgt_200_anom(0,:,:), u_200_r_p1)
copy_VarCoords(hgt_200_anom(0,:,:), u_200_r_p2)
copy_VarCoords(hgt_200_anom(0,:,:), u_200_r_p3)
copy_VarCoords(hgt_200_anom(0,:,:), u_850_r_p1)
copy_VarCoords(hgt_200_anom(0,:,:), u_850_r_p2)
copy_VarCoords(hgt_200_anom(0,:,:), u_850_r_p3)
copy_VarCoords(hgt_200_anom(0,:,:), v_200_r_p1)
copy_VarCoords(hgt_200_anom(0,:,:), v_200_r_p2)
copy_VarCoords(hgt_200_anom(0,:,:), v_200_r_p3)
copy_VarCoords(hgt_200_anom(0,:,:), v_850_r_p1)
copy_VarCoords(hgt_200_anom(0,:,:), v_850_r_p2)
copy_VarCoords(hgt_200_anom(0,:,:), v_850_r_p3)
copy_VarCoords(hgt_200_anom(0,:,:), t2m_r_p1)
copy_VarCoords(hgt_200_anom(0,:,:), t2m_r_p2)
copy_VarCoords(hgt_200_anom(0,:,:), t2m_r_p3)

;>---------------------------------------------------------------------------------------<
;;                    t-test
;>---------------------------------------------------------------------------------------<

confi_h200_p1 = confi(hgt_200_r_p1)
confi_h200_p2 = confi(hgt_200_r_p2)
confi_h200_p3 = confi(hgt_200_r_p3)
confi_h850_p1 = confi(hgt_850_r_p1)
confi_h850_p2 = confi(hgt_850_r_p2)
confi_h850_p3 = confi(hgt_850_r_p3)
confi_t2m_p1 = confi(t2m_r_p1)
confi_t2m_p2 = confi(t2m_r_p2)
confi_t2m_p3 = confi(t2m_r_p3)

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
res@tmBorderThicknessF = 1.5
; res@gsnStringFont = 22

res_sig = res
res_tn = res
res850 = res
res_pre = res
res_uv = res
res2 = res

res@vpWidthF = 0.8
res@vpHeightF = 0.4
res@mpShapeMode = "FreeAspect"
res@mpMinLatF = 10
res@mpMaxLatF = 90
res@mpMinLonF = 140
res@mpMaxLonF = 310
res@mpCenterLonF = 225
res@mpFillOn = False
res@mpGeophysicalLineColor = "gray40"
res@mpGeophysicalLineThicknessF = 1.0
res@gsnLeftStringOrthogonalPosF = 0.01
res@gsnLeftStringFontHeightF = 35
res@gsnRightStringOrthogonalPosF = 0.01
res@gsnRightStringFontHeightF = 35

res@tmXBMinorLengthF = 0
res@tmYLMinorLengthF = 0
res@tmXBMajorLengthF = 0.005
res@tmYLMajorLengthF = 0.005
res@tmXBLabelDeltaF = -0.5
res@tmYLLabelDeltaF = -0.5
res@tmXBLabelFontThicknessF = 1.5
res@tmYLLabelFontThicknessF = 1.5
res@tmXBLabelFontHeightF = 0.027
res@tmYLLabelFontHeightF = 0.027
res@tmXBTickSpacingF = 30
res@tmYLTickSpacingF = 20
res@tmXBMajorThicknessF = 1.5
res@tmYLMajorThicknessF = 1.5
res@tmXTOn = False
res@tmYROn = False

resh = res
res_sst = res
rest = res

resh@cnLinesOn = False
resh@cnLineLabelsOn = False
resh@cnInfoLabelOn = False
resh@cnFillOn = True
resh@cnLevelSelectionMode = "ExplicitLevels"
resh@cnFillPalette = "MPL_BrBG"
resh@cnFillColors = (/100,95,90,85,80,-1,-1,45,40,35,30,25/)

resh@lbLabelBarOn = True
resh@lbOrientation = "Vertical"
resh@pmLabelBarWidthF = 0.06
resh@lbLabelFontHeightF = 0.025
resh@lbBoxEndCapStyle = "TriangleBothEnds"


res_uv@vcGlyphStyle = "CurlyVector"
res_uv@vcMinDistanceF = 0.03
res_uv@vcLineArrowThicknessF = 0.8
; res_uv@vcLineArrowColor = "grey"

res_uv@vcRefAnnoOn = True
res_uv@vcRefLengthF = 0.035
res_uv@vcRefAnnoBackgroundColor = "white"
res_uv@vcRefAnnoPerimOn = True   ;关闭边框
res_uv@vcRefAnnoFontHeightF = 0.012
res_uv@vcRefAnnoString1On =   False ;设置标注箭头上方不显示注释
res_uv@vcRefAnnoString2On = True   ;设置标注箭头下方显示注释
res_uv@vcRefAnnoFontThicknessF = 1.5
res_uv@vcRefAnnoOrthogonalPosF = -0.15
res_uv@vcRefAnnoParallelPosF = 0.999
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

res_tn@vcGlyphStyle = "CurlyVector"
res_tn@vcMinDistanceF = 0.023
res_tn@vcLineArrowThicknessF = 1
res_tn@vcLineArrowColor = "purple3"
res_tn@vcLineArrowHeadMaxSizeF = 0.015
res_tn@vcLineArrowHeadMinSizeF = 0.015


res_tn@vcRefAnnoOn = True
res_tn@vcRefLengthF = 0.035
res_tn@vcRefAnnoBackgroundColor = "white"
res_tn@vcRefAnnoPerimOn = True   ;关闭边框
res_tn@vcRefAnnoFontHeightF = 0.012
res_tn@vcRefAnnoString1On =   False ;设置标注箭头上方不显示注释
res_tn@vcRefAnnoString2On = True   ;设置标注箭头下方显示注释
res_tn@vcRefAnnoFontThicknessF = 1.5

res_tn@vcRefAnnoOrthogonalPosF = -0.21
res_tn@vcRefAnnoParallelPosF = 0.999
res_tn@vcVectorDrawOrder = "PostDraw"


minlat = 65.
maxlat = 85.
minlon = -150.
maxlon = -60.

rest@gsnLeftStringOrthogonalPosF = 0.1
rest@vpWidthF = 0.7
rest@vpHeightF = 0.45
rest@mpProjection = "LambertConformal"
rest@mpMinLatF := minlat
rest@mpMaxLatF := maxlat
rest@mpMinLonF := minlon
rest@mpMaxLonF := maxlon
rest@gsnMaskLambertConformal = True
rest@mpGridAndLimbOn = True 
rest@mpGridLineDashPattern = 2 
rest@mpGridLatSpacingF = 5
rest@mpGridLonSpacingF = 20

rest@cnLinesOn = False
rest@cnLineLabelsOn = False
rest@cnInfoLabelOn = False
rest@cnFillOn = True
rest@cnFillPalette = "BlWhRe"
rest@cnLevelSelectionMode = "ExplicitLevels"
rest@lbLabelBarOn = True
rest@lbOrientation = "Vertical"
rest@pmLabelBarWidthF = 0.06
rest@lbLabelFontHeightF = 0.025
rest@lbBoxEndCapStyle = "TriangleBothEnds"
rest@pmLabelBarOrthogonalPosF = 0.1

;;;>-----------------------------------------------------------------------------------------<

wks = gsn_open_wks("pdf", "./figs5")
    
;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

resh@gsnLeftString = "(a) Reg_UVZ250 onto IPDC"
resh@cnLevels := (/-15,-12,-9,-6,-3,0,3,6,9,12,15/)
plot_h200_p1 = gsn_csm_contour_map(wks, hgt_200_r_p1, resh)
plot_h200_sig_p1 = gsn_csm_contour(wks, confi_h200_p1, res_sig)
overlay(plot_h200_p1, plot_h200_sig_p1)

resh@gsnLeftString = "(d) Reg_UVZ250 onto IPDC"
plot_h200_p2 = gsn_csm_contour_map(wks, hgt_200_r_p2, resh)
plot_h200_sig_p2 = gsn_csm_contour(wks, confi_h200_p2, res_sig)
overlay(plot_h200_p2, plot_h200_sig_p2)

resh@gsnLeftString = "(g) Reg_UVZ250 onto IPDC"
plot_h200_p3 = gsn_csm_contour_map(wks, hgt_200_r_p3, resh)
plot_h200_sig_p3 = gsn_csm_contour(wks, confi_h200_p3, res_sig)
overlay(plot_h200_p3, plot_h200_sig_p3)

;;;>-----------------------------------------------------------------------------------------<

resh@gsnLeftString = "(b) Reg_UVZ850 onto IPDC"
resh@cnLevels := (/-10,-7,-5,-3,-1,0,1,3,5,7,10/)
plot_h850_p1 = gsn_csm_contour_map(wks, hgt_850_r_p1, resh)
plot_h850_sig_p1 = gsn_csm_contour(wks, confi_h850_p1, res_sig)
overlay(plot_h850_p1, plot_h850_sig_p1)

resh@gsnLeftString = "(e) Reg_UVZ850 onto IPDC"
plot_h850_p2 = gsn_csm_contour_map(wks, hgt_850_r_p2, resh)
plot_h850_sig_p2 = gsn_csm_contour(wks, confi_h850_p2, res_sig)
overlay(plot_h850_p2, plot_h850_sig_p2)

resh@gsnLeftString = "(h) Reg_UVZ850 onto IPDC"
plot_h850_p3 = gsn_csm_contour_map(wks, hgt_850_r_p3, resh)
plot_h850_sig_p3 = gsn_csm_contour(wks, confi_h850_p3, res_sig)
overlay(plot_h850_p3, plot_h850_sig_p3)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

res_uv@vcRefMagnitudeF := 1
res_uv@vcRefAnnoString2 := "1 m/s"
min200 = 0.3     ;;  mask小值  raw

u_200_r_p1 = where(sqrt((/u_200_r_p1/)^2 + (/v_200_r_p1/)^2) .gt. min200, u_200_r_p1, 0)
v_200_r_p1 = where(sqrt((/u_200_r_p1/)^2 + (/v_200_r_p1/)^2) .gt. min200, v_200_r_p1, 0)

u_200_r_p2 = where(sqrt((/u_200_r_p2/)^2 + (/v_200_r_p2/)^2) .gt. min200, u_200_r_p2, 0)
v_200_r_p2 = where(sqrt((/u_200_r_p2/)^2 + (/v_200_r_p2/)^2) .gt. min200, v_200_r_p2, 0)

u_200_r_p3 = where(sqrt((/u_200_r_p3/)^2 + (/v_200_r_p3/)^2) .gt. min200, u_200_r_p3, 0)
v_200_r_p3 = where(sqrt((/u_200_r_p3/)^2 + (/v_200_r_p3/)^2) .gt. min200, v_200_r_p3, 0)

plot_uv200_p1 = gsn_csm_vector(wks, u_200_r_p1, v_200_r_p1, res_uv)
overlay(plot_h200_p1, plot_uv200_p1)

plot_uv200_p2 = gsn_csm_vector(wks, u_200_r_p2, v_200_r_p2, res_uv)
overlay(plot_h200_p2, plot_uv200_p2)

plot_uv200_p3 = gsn_csm_vector(wks, u_200_r_p3, v_200_r_p3, res_uv)
overlay(plot_h200_p3, plot_uv200_p3)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

res_uv@vcRefMagnitudeF := 0.8
res_uv@vcRefAnnoString2 := "0.8 m/s"
min850 = 0.2     ;;  mask小值  raw

u_850_r_p1 = where(sqrt((/u_850_r_p1/)^2 + (/v_850_r_p1/)^2) .gt. min850, u_850_r_p1, 0)
v_850_r_p1 = where(sqrt((/u_850_r_p1/)^2 + (/v_850_r_p1/)^2) .gt. min850, v_850_r_p1, 0)

u_850_r_p2 = where(sqrt((/u_850_r_p2/)^2 + (/v_850_r_p2/)^2) .gt. min850, u_850_r_p2, 0)
v_850_r_p2 = where(sqrt((/u_850_r_p2/)^2 + (/v_850_r_p2/)^2) .gt. min850, v_850_r_p2, 0)

u_850_r_p3 = where(sqrt((/u_850_r_p3/)^2 + (/v_850_r_p3/)^2) .gt. min850, u_850_r_p3, 0)
v_850_r_p3 = where(sqrt((/u_850_r_p3/)^2 + (/v_850_r_p3/)^2) .gt. min850, v_850_r_p3, 0)

plot_uv850_p1 = gsn_csm_vector(wks, u_850_r_p1, v_850_r_p1, res_uv)
overlay(plot_h850_p1, plot_uv850_p1)

plot_uv850_p2 = gsn_csm_vector(wks, u_850_r_p2, v_850_r_p2, res_uv)
overlay(plot_h850_p2, plot_uv850_p2)

plot_uv850_p3 = gsn_csm_vector(wks, u_850_r_p3, v_850_r_p3, res_uv)
overlay(plot_h850_p3, plot_uv850_p3)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

gres = True
gres@gsLineColor = "blue"
gres@gsLineThicknessF = 2

lonw1 = 230
; lonw1 = 215
lone1 = 280
lats1 = 72
latn1 = 80
x2 = (/lonw1,lone1,lone1,lonw1,lonw1/)
y2 = (/lats1,lats1,latn1,latn1,lats1/)

lat_spacing = 5   ; use integers here
lon_spacing = 20

rest@cnLevels := (/-1,-0.8,-0.6, -0.4, -0.2,0, 0.2, 0.4, 0.6, 0.8, 1/)
rest@cnFillColors = (/30,33,36,39,42,-1,-1,58,61,66,70,80/)

rest@gsnLeftString = "(c) Reg_SAT onto IPDC"
plot_t2m_p1 = gsn_csm_contour_map(wks, t2m_r_p1, rest)
plot_sig_t2m_p1 = gsn_csm_contour(wks, confi_t2m_p1, res_sig)
overlay(plot_t2m_p1, plot_sig_t2m_p1)
dum1 = gsn_add_polyline(wks, plot_sig_t2m_p1, x2, y2, gres)
add_lc_labels(wks, plot_sig_t2m_p1, minlat, maxlat, minlon, maxlon, lat_spacing, lon_spacing)

rest@gsnLeftString = "(f) Reg_SAT onto IPDC"
plot_t2m_p2 = gsn_csm_contour_map(wks, t2m_r_p2, rest)
plot_sig_t2m_p2 = gsn_csm_contour(wks, confi_t2m_p2, res_sig)
overlay(plot_t2m_p2, plot_sig_t2m_p2)
dum2 = gsn_add_polyline(wks, plot_sig_t2m_p2, x2, y2, gres)
add_lc_labels(wks, plot_sig_t2m_p2, minlat, maxlat, minlon, maxlon, lat_spacing, lon_spacing)

rest@gsnLeftString = "(i) Reg_SAT onto IPDC"
plot_t2m_p3 = gsn_csm_contour_map(wks, t2m_r_p3, rest)
plot_sig_t2m_p3 = gsn_csm_contour(wks, confi_t2m_p3, res_sig)
overlay(plot_t2m_p3, plot_sig_t2m_p3)
dum3 = gsn_add_polyline(wks, plot_sig_t2m_p3, x2, y2, gres)
add_lc_labels(wks, plot_sig_t2m_p3, minlat, maxlat, minlon, maxlon, lat_spacing, lon_spacing)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

pres = True
pres@gsnPanelXWhiteSpacePercent = 3
pres@gsnPanelYWhiteSpacePercent = 7
gsn_panel(wks, (/plot_h200_p1, plot_h200_p2, plot_h200_p3, \
                plot_h850_p1, plot_h850_p2, plot_h850_p3, \
                plot_t2m_p1, plot_t2m_p2, plot_t2m_p3/), (/3, 3/), pres)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

end
