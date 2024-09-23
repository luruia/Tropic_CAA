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

; undef tnwaf(var)
function tnwaf(hgt_r, u_clm, v_clm)     ;;;   返回t检验结果
    local zvarorg1, cuvar, cvvar, level, gc, ga, re, sclhgt, pi, f, ilat, nlat, lon, lat, nlon, coslat, \
            coslattmp, ftmp, cumag, psidev, dpsidevdlon, ddpsidevdlonlon, dpsidevdlat, ddpsidevdlonlat, ddpsidevdlatlat, \
            xuterm, xvterm, yvterm, fx, fy
    begin

        zvarorg1 = hgt_r   ;;  异常场

        cuvar = u_clm     ;;   气候态
        cvvar = v_clm

        ; lon = hgt_r&g0_lon_3
        ; lat = hgt_r&g0_lat_2
        lon = hgt_r&longitude
        lat = hgt_r&latitude
        nlon = dimsizes(lon)
        nlat = dimsizes(lat)

        level = 250

        ;;;>-----------------------------------------------------------------------------------------<

        gc = 290   ;;  Gas constant
        ga = 9.80665   ;;  Gravitational acceleration
        re = 6378388    ;;  Radius of the earth
        sclhgt = 8000.   ;; scale height
        pi = atan(1.0)*4.   ;; pi

        f =  2.*2.*pi/(60.*60.*24.)*sin(pi/180. * lat(:))   ;; Coriolis parameter
        f!0 = "lat"
        f&lat = lat
        f@_FillValue = doubletofloat(zvarorg1@_FillValue)

        ; missing for 10S - 10N
        do ilat = 0, nlat-1
            if (abs(lat(ilat) ).lt. 35. ) then
                f(ilat)= f@_FillValue
            end if
        end do


        coslat = cos(lat(:)*pi/180.)   ;; cosine

        coslattmp = conform_dims(dimsizes(zvarorg1), coslat, 0)
        ftmp = conform_dims(dimsizes(zvarorg1), f, 0)

        ; magnitude of climatological wind
        cumag = sqrt(cuvar^2 + cvvar^2)
        cumag@_FillValue = cuvar@_FillValue
        cumag = where(cumag .gt. 0, cumag, cumag@_FillValue)
        copy_VarCoords(cuvar,cumag)

        ; QG steam function for anomaly
        psidev = zvarorg1*ga /ftmp
        copy_VarCoords(zvarorg1, psidev)

        ;dpsidev/dlon
        dpsidevdlon = center_finite_diff_n(psidev, lon*pi/180., True, 0, 1)

        ;ddpsidev/dlonlon
        ddpsidevdlonlon = center_finite_diff_n(dpsidevdlon, lon*pi/180., True, 0, 1)

        ;dpsidev/dlat
        dpsidevdlat = center_finite_diff_n(psidev, lat*pi/180., False, 0, 0)

        ;ddpsidev/dlonlat
        ddpsidevdlonlat = center_finite_diff_n(dpsidevdlon, lat*pi/180., False, 0, 0)

        ;ddpsidev/dlatdlat
        ddpsidevdlatlat = center_finite_diff_n(dpsidevdlat, lat*pi/180., False, 0, 0)

        xuterm = (dpsidevdlon*dpsidevdlon - psidev*ddpsidevdlonlon)
        xvterm = (dpsidevdlon*dpsidevdlat - psidev*ddpsidevdlonlat)
        ;yuterm = xvterm
        yvterm = (dpsidevdlat*dpsidevdlat - psidev*ddpsidevdlatlat)


        ;x-component of (38)
        fx = level/1000./(2.*cumag*re*re)*( cuvar/coslattmp * xuterm + cvvar * xvterm)

        ;y-component 
        fy = level/1000./(2.*cumag*re*re)*( cuvar*xvterm + coslattmp*cvvar*yvterm)

        ; for output
        fx!0 = "lat"
        fx&lat = lat
        fx!1 = "lon"
        fx&lon = lon

        fy!0 = "lat"
        fy&lat = lat
        fy!1 = "lon"
        fy&lon = lon

        fx@units = "m^2/s^2"
        fy@units = "m^2/s^2"

        return [/fx, fy/]
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

f_u200 = addfile("./u250_index_ea_170.nc", "r")
pc_u200 = f_u200->pc
delete(f_u200)

ind_pos = ind(pc_u200 .gt. 0.5)
ind_neg = ind(pc_u200 .lt. -0.5)

;;;>-----------------------------------------------------------------------------------------<
;;;                               读取数据
;;;>-----------------------------------------------------------------------------------------<

f_pc = addfile("./olr_index.nc", "r")
pc = f_pc->pc(yyyymm_s/100-1979:yyyymm_e/100-1979)
pc = pc * (-1)

pc_pos = pc(ind_pos)
pc_neg = pc(ind_neg)
pc_pos = dim_standardize(pc_pos, 1)
pc_neg = dim_standardize(pc_neg, 1)

;;;>-----------------------------------------------------------------------------------------<

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

hgt_200_r_pos = regCoef_n(pc_pos, hgt_200_anom(ind_pos,:,:), 0, 0)    
hgt_200_r_neg = regCoef_n(pc_neg, hgt_200_anom(ind_neg,:,:), 0, 0)    
hgt_850_r_pos = regCoef_n(pc_pos, hgt_850_anom(ind_pos,:,:), 0, 0)    
hgt_850_r_neg = regCoef_n(pc_neg, hgt_850_anom(ind_neg,:,:), 0, 0)    
u_200_r_pos = regCoef_n(pc_pos, u_200_anom(ind_pos,:,:), 0, 0)    
u_200_r_neg = regCoef_n(pc_neg, u_200_anom(ind_neg,:,:), 0, 0)    
u_850_r_pos = regCoef_n(pc_pos, u_850_anom(ind_pos,:,:), 0, 0)    
u_850_r_neg = regCoef_n(pc_neg, u_850_anom(ind_neg,:,:), 0, 0)    
v_200_r_pos = regCoef_n(pc_pos, v_200_anom(ind_pos,:,:), 0, 0)    
v_200_r_neg = regCoef_n(pc_neg, v_200_anom(ind_neg,:,:), 0, 0)    
v_850_r_pos = regCoef_n(pc_pos, v_850_anom(ind_pos,:,:), 0, 0)    
v_850_r_neg = regCoef_n(pc_neg, v_850_anom(ind_neg,:,:), 0, 0)    
t2m_r_pos = regCoef_n(pc_pos, t2m_anom(ind_pos,:,:), 0, 0)    
t2m_r_neg = regCoef_n(pc_neg, t2m_anom(ind_neg,:,:), 0, 0)    

copy_VarCoords(hgt_200_anom(0,:,:), hgt_200_r_pos)
copy_VarCoords(hgt_200_anom(0,:,:), hgt_200_r_neg)
copy_VarCoords(hgt_200_anom(0,:,:), hgt_850_r_pos)
copy_VarCoords(hgt_200_anom(0,:,:), hgt_850_r_neg)
copy_VarCoords(hgt_200_anom(0,:,:), u_200_r_pos)
copy_VarCoords(hgt_200_anom(0,:,:), u_200_r_neg)
copy_VarCoords(hgt_200_anom(0,:,:), u_850_r_pos)
copy_VarCoords(hgt_200_anom(0,:,:), u_850_r_neg)
copy_VarCoords(hgt_200_anom(0,:,:), v_200_r_pos)
copy_VarCoords(hgt_200_anom(0,:,:), v_200_r_neg)
copy_VarCoords(hgt_200_anom(0,:,:), v_850_r_pos)
copy_VarCoords(hgt_200_anom(0,:,:), v_850_r_neg)
copy_VarCoords(hgt_200_anom(0,:,:), t2m_r_pos)
copy_VarCoords(hgt_200_anom(0,:,:), t2m_r_neg)

;>---------------------------------------------------------------------------------------<
;;                    t-test
;>---------------------------------------------------------------------------------------<

confi_h200_pos = confi(hgt_200_r_pos)
confi_h200_neg = confi(hgt_200_r_neg)
confi_h850_pos = confi(hgt_850_r_pos)
confi_h850_neg = confi(hgt_850_r_neg)
confi_t2m_pos = confi(t2m_r_pos)
confi_t2m_neg = confi(t2m_r_neg)

;>---------------------------------------------------------------------------------------<
;;                    WAF
;>---------------------------------------------------------------------------------------<

tn_pos = tnwaf(hgt_200_r_pos, u_200_clim, v_200_clim)
fx_pos = tn_pos[0]
fy_pos = tn_pos[1]

tn_neg = tnwaf(hgt_200_r_neg, u_200_clim, v_200_clim)
fx_neg = tn_neg[0]
fy_neg = tn_neg[1]

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

res_uv@vcRefAnnoOn = True
res_uv@vcRefLengthF = 0.035
res_uv@vcRefAnnoBackgroundColor = "white"
res_uv@vcRefAnnoPerimOn = True   ;关闭边框
res_uv@vcRefAnnoFontHeightF = 0.012
res_uv@vcRefAnnoString1On =   False ;设置标注箭头上方不显示注释
res_uv@vcRefAnnoString2On = True   ;设置标注箭头下方显示注释
res_uv@vcRefAnnoFontThicknessF = 1.5
res_uv@vcRefAnnoOrthogonalPosF = -0.16
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
res_tn@vcLineArrowThicknessF = 0.8
res_tn@vcLineArrowColor = "purple"
res_tn@vcLineArrowHeadMaxSizeF = 0.01
res_tn@vcLineArrowHeadMinSizeF = 0.01


res_tn@vcRefAnnoOn = True
res_tn@vcRefLengthF = 0.035
res_tn@vcRefAnnoBackgroundColor = "white"
res_tn@vcRefAnnoPerimOn = True   ;关闭边框
res_tn@vcRefAnnoFontHeightF = 0.008
res_tn@vcRefAnnoString1On =   False ;设置标注箭头上方不显示注释
res_tn@vcRefAnnoString2On = True   ;设置标注箭头下方显示注释
res_tn@vcRefAnnoFontThicknessF = 1.5

res_tn@vcRefAnnoOrthogonalPosF = -0.127
res_tn@vcRefAnnoParallelPosF = 0.08
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

wks = gsn_open_wks("pdf", "./figs8")
    
;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

resh@gsnLeftString = "(d) Reg_Z250&WAF in strong jet"
resh@cnLevels := (/-15,-12,-9,-6,-3,0,3,6,9,12,15/)
; res@cnLevels := (/-20,-15,-10,-5,5,10,15,20/)
plot_h200_pos = gsn_csm_contour_map(wks, hgt_200_r_pos, resh)
; plot_h200 = ColorNegDashZeroPosContour(plot_h200, "blue", "black", "red")
plot_h200_sig_pos = gsn_csm_contour(wks, confi_h200_pos, res_sig)
overlay(plot_h200_pos, plot_h200_sig_pos)

resh@gsnLeftString = "(a) Reg_Z250&WAF in weak jet"
plot_h200_neg = gsn_csm_contour_map(wks, hgt_200_r_neg, resh)
plot_h200_sig_neg = gsn_csm_contour(wks, confi_h200_neg, res_sig)
overlay(plot_h200_neg, plot_h200_sig_neg)

;;;>-----------------------------------------------------------------------------------------<

resh@gsnLeftString = "(e) Reg_UVZ850 in strong jet"
resh@cnLevels := (/-10,-7,-5,-3,-1,0,1,3,5,7,10/)
; res@cnLevels := (/-10,-8,-6,-4,-2,2,4,6,8,10/)
plot_h850_pos = gsn_csm_contour_map(wks, hgt_850_r_pos, resh)
; plot_h200 = ColorNegDashZeroPosContour(plot_h200, "blue", "black", "red")
plot_h850_sig_pos = gsn_csm_contour(wks, confi_h850_pos, res_sig)
overlay(plot_h850_pos, plot_h850_sig_pos)

resh@gsnLeftString = "(b) Reg_UVZ850 in weak jet"
plot_h850_neg = gsn_csm_contour_map(wks, hgt_850_r_neg, resh)
plot_h850_sig_neg = gsn_csm_contour(wks, confi_h850_neg, res_sig)
overlay(plot_h850_neg, plot_h850_sig_neg)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

res_tn@vcRefMagnitudeF = 0.1
res_tn@vcRefAnnoString2 = "0.1 m~S~2~N~/s~S~2~N~"
minf = 0.03 

fx_pos = where(sqrt((/fx_pos/)^2 + (/fy_pos/)^2) .gt. minf, fx_pos, 0)
fy_pos = where(sqrt((/fx_pos/)^2 + (/fy_pos/)^2) .gt. minf, fy_pos, 0)

fx_neg = where(sqrt((/fx_neg/)^2 + (/fy_neg/)^2) .gt. minf, fx_neg, 0)
fy_neg = where(sqrt((/fx_neg/)^2 + (/fy_neg/)^2) .gt. minf, fy_neg, 0)

plot_tn_pos = gsn_csm_vector(wks, fx_pos, fy_pos, res_tn)
overlay(plot_h200_pos, plot_tn_pos)

plot_tn_neg = gsn_csm_vector(wks, fx_neg, fy_neg, res_tn)
overlay(plot_h200_neg, plot_tn_neg)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

res_uv@vcRefMagnitudeF := 1.0
res_uv@vcRefAnnoString2 := "1.0 m/s"
min850 = 0.2     ;;  mask小值  raw

u_850_r_pos = where(sqrt((/u_850_r_pos/)^2 + (/v_850_r_pos/)^2) .gt. min850, u_850_r_pos, 0)
v_850_r_pos = where(sqrt((/u_850_r_pos/)^2 + (/v_850_r_pos/)^2) .gt. min850, v_850_r_pos, 0)

u_850_r_neg = where(sqrt((/u_850_r_neg/)^2 + (/v_850_r_neg/)^2) .gt. min850, u_850_r_neg, 0)
v_850_r_neg = where(sqrt((/u_850_r_neg/)^2 + (/v_850_r_neg/)^2) .gt. min850, v_850_r_neg, 0)

plot_uv850_pos = gsn_csm_vector(wks, u_850_r_pos, v_850_r_pos, res_uv)
overlay(plot_h850_pos, plot_uv850_pos)

plot_uv850_neg = gsn_csm_vector(wks, u_850_r_neg, v_850_r_neg, res_uv)
overlay(plot_h850_neg, plot_uv850_neg)

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

rest@gsnLeftString = "(f) Reg_SAT in strong jet"
plot_t2m_pos = gsn_csm_contour_map(wks, t2m_r_pos, rest)
plot_sig_t2m_pos = gsn_csm_contour(wks, confi_t2m_pos, res_sig)
overlay(plot_t2m_pos, plot_sig_t2m_pos)
dum1 = gsn_add_polyline(wks, plot_sig_t2m_pos, x2, y2, gres)
add_lc_labels(wks, plot_sig_t2m_pos, minlat, maxlat, minlon, maxlon, lat_spacing, lon_spacing)

rest@gsnLeftString = "(c) Reg_SAT in weak jet"
plot_t2m_neg = gsn_csm_contour_map(wks, t2m_r_neg, rest)
plot_sig_t2m_neg = gsn_csm_contour(wks, confi_t2m_neg, res_sig)
overlay(plot_t2m_neg, plot_sig_t2m_neg)
dum2 = gsn_add_polyline(wks, plot_sig_t2m_neg, x2, y2, gres)
add_lc_labels(wks, plot_sig_t2m_neg, minlat, maxlat, minlon, maxlon, lat_spacing, lon_spacing)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

pres = True
pres@gsnPanelXWhiteSpacePercent = 3
pres@gsnPanelYWhiteSpacePercent = 5
gsn_panel(wks, (/plot_h200_neg, plot_h200_pos, \
                plot_h850_neg, plot_h850_pos, \
                plot_t2m_neg, plot_t2m_pos/), (/3, 2/), pres)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

end
