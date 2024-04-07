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
      txres@txFontHeightF = 0.02
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
        lat_label_rgt = "          "  + lat_values(n) + "~S~o~N~"

    ;---Check if North, South, or Zero
        if(lat_values(n).lt.0) then
          lat_label_lft = lat_values(n) + "~S~o~N~S          "
          lat_label_rgt = lat_label_rgt + "S"
        end if
        if(lat_values(n).gt.0) then
          lat_label_lft = lat_values(n) + "~S~o~N~N          "
          lat_label_rgt = lat_label_rgt + "N"
        end if
        if(lat_values(n).eq.0) then
          lat_label_lft = lat_values(n) + "~S~o~N~           "
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
        lon_label_bot = "  ~C~  ~C~" + abs(lon_values(n)) + "~S~o~N~"
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
        var_season = dtrend_msg_n(var_season&time, var_season, False, False, 0)

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
        lon = hgt_r&lon
        lat = hgt_r&lat
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

begin

;;;>-----------------------------------------------------------------------------------------<
;;;                                时间段
;;;>-----------------------------------------------------------------------------------------<

yyyymm_s = 197901
yyyymm_e = 202112

;;;>-----------------------------------------------------------------------------------------<
;;;                               读取数据
;;;>-----------------------------------------------------------------------------------------<

; f_sst = addfile("./sst_index_polar.nc", "r")
; pc_sst = f_sst->pc
; delete(f_sst)
; pc = dim_standardize(-pc_sst, 1)

f_sst = addfile("./olr_io.nc", "r")
pc_sst = f_sst->pc
delete(f_sst)
pc = dim_standardize(-pc_sst, 1)


f4 = addfile("/mnt/d/data/1-initial/reanalysis/era5/monthly/ERA5_U_1x1_1940_2022_mon.nc", "r")    ;;  风场
time = cd_calendar(f4->time, 1) 
ind_s = ind(time .eq. yyyymm_s)
ind_e = ind(time .eq. yyyymm_e)
u_200 = short2flt(f4->u(ind_s:ind_e,{250},:,:)) 
delete([/time, ind_s, ind_e, f4/])


f1 = addfile("./echam5/echam_197901-202112.nc", "r") 
f2 = addfile("./echam5/echam_197901-202112_pos.nc", "r") 
time = cd_calendar(f1->time, 1) 
ind_s = ind(time .eq. yyyymm_s)
ind_e = ind(time .eq. yyyymm_e)

hgt_200_ctrl = f1->geopoth(ind_s:ind_e,{25000},:,:)
u_200_ctrl = f1->u(ind_s:ind_e,{25000},:,:)
v_200_ctrl = f1->v(ind_s:ind_e,{25000},:,:) 
t2m_ctrl = f1->temp2(ind_s:ind_e,:,:)
p1_ctrl = f1->precip(ind_s:ind_e,:,:)
pre_ctrl = p1_ctrl * 24*30*30.
copy_VarMeta(p1_ctrl, pre_ctrl)
delete(p1_ctrl)

hgt_200_sens = f2->geopoth(ind_s:ind_e,{25000},:,:)
u_200_sens = f2->u(ind_s:ind_e,{25000},:,:)
v_200_sens = f2->v(ind_s:ind_e,{25000},:,:) 
t2m_sens = f2->temp2(ind_s:ind_e,:,:)
p1_sens = f2->precip(ind_s:ind_e,:,:)
pre_sens = p1_sens * 24*30*30.
copy_VarMeta(p1_sens, pre_sens)
delete(p1_sens)

lon = f1->lon
lat = f1->lat
nlon = dimsizes(lon)
nlat = dimsizes(lat)
dims_uvz = dimsizes(hgt_200_ctrl)

delete([/time, ind_s, ind_e, f1/])

;;;>-----------------------------------------------------------------------------------------<
;;;                                  季节平均 距平
;;;>-----------------------------------------------------------------------------------------<

n_year = dims_uvz(0)/12

hgt_200_anom_ctrl = to_season(hgt_200_ctrl)
pre_anom_ctrl = to_season(pre_ctrl)
t2m_anom_ctrl = to_season(t2m_ctrl)
u_200_anom_ctrl = to_season(u_200_ctrl)

u_200_clim_ctrl = to_clim(u_200_ctrl)
v_200_clim_ctrl = to_clim(v_200_ctrl)

delete([/hgt_200_ctrl, u_200_ctrl, v_200_ctrl, pre_ctrl, t2m_ctrl/])

;;;>-----------------------------------------------------------------------------------------<

hgt_200_anom_sens = to_season(hgt_200_sens)
pre_anom_sens = to_season(pre_sens)
t2m_anom_sens = to_season(t2m_sens)
u_200_anom_sens = to_season(u_200_sens)

u_200_clim_sens = to_clim(u_200_sens)
v_200_clim_sens = to_clim(v_200_sens)

delete([/hgt_200_sens, u_200_sens, v_200_sens, pre_sens, t2m_sens/])

;;;>-----------------------------------------------------------------------------------------<

u_200_anom = to_season(u_200)
u_200_clim = to_clim(u_200)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

u_200_diff = u_200_clim_ctrl - u_200_clim_sens
confi_u_200_diff = comp_sig(u_200_anom_ctrl, u_200_anom_sens)
copy_VarCoords(u_200_clim_sens, u_200_diff)
copy_VarCoords(u_200_clim_sens, confi_u_200_diff)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

lonw21 = 75
lone21 = 135
lats21 = -15
latn21 = 15
; x1 = (/lonw,lone,lone,lonw,lonw/)
; y1 = (/lats,lats,latn,latn,lats/)
pre_mc_ctrl = dim_avg_n_Wrap(pre_anom_ctrl(:,{lats21:latn21},{lonw21:lone21}), (/1, 2/))
pre_mc_ctrl = dim_standardize(pre_mc_ctrl, 1)
pre_mc_sens = dim_avg_n_Wrap(pre_anom_sens(:,{lats21:latn21},{lonw21:lone21}), (/1, 2/))
pre_mc_sens = dim_standardize(pre_mc_sens, 1)

lonw22 = 150
lone22 = 270
lats22 = -15
latn22 = 15
; x3 = (/lonw1,lone1,lone1,lonw1,lonw1/)
; y3 = (/lats1,lats1,latn1,latn1,lats1/)
pre_wp_ctrl = dim_avg_n_Wrap(pre_anom_ctrl(:,{lats22:latn22},{lonw22:lone22}), (/1, 2/))
pre_wp_ctrl = dim_standardize(pre_wp_ctrl, 1)
pre_wp_sens = dim_avg_n_Wrap(pre_anom_sens(:,{lats22:latn22},{lonw22:lone22}), (/1, 2/))
pre_wp_sens = dim_standardize(pre_wp_sens, 1)

pc_ctrl = pre_mc_ctrl - pre_wp_ctrl
pc_ctrl = dim_standardize(pc_ctrl, 1)
pc_sens = pre_mc_sens - pre_wp_sens
pc_sens = dim_standardize(pc_sens, 1)

;;;>-----------------------------------------------------------------------------------------<
;;;                                      回归
;;;>-----------------------------------------------------------------------------------------<

 
hgt_200_r_ctrl = regCoef_n(pc_ctrl, hgt_200_anom_ctrl, 0, 0)  
t2m_r_ctrl = regCoef_n(pc_ctrl, t2m_anom_ctrl, 0, 0)  
copy_VarCoords(t2m_anom_ctrl(0,:,:), t2m_r_ctrl)
copy_VarCoords(hgt_200_anom_ctrl(0,:,:), hgt_200_r_ctrl)

hgt_200_r_sens = regCoef_n(pc_sens, hgt_200_anom_sens, 0, 0)  
t2m_r_sens = regCoef_n(pc_sens, t2m_anom_sens, 0, 0)  
copy_VarCoords(t2m_anom_ctrl(0,:,:), t2m_r_sens)
copy_VarCoords(hgt_200_anom_ctrl(0,:,:), hgt_200_r_sens)

;>---------------------------------------------------------------------------------------<

u_200_r = regCoef_n(pc, u_200_anom, 0, 0)  
copy_VarCoords(u_200_anom(0,:,:), u_200_r)

;>---------------------------------------------------------------------------------------<
;;                    t-test
;>---------------------------------------------------------------------------------------<

confi_h200_ctrl = confi(hgt_200_r_ctrl)
confi_h200_sens = confi(hgt_200_r_sens)
confi_t2m_ctrl = confi(t2m_r_ctrl)
confi_t2m_sens = confi(t2m_r_sens)

;;;>-----------------------------------------------------------------------------------------<

confi_u200 = confi(u_200_r)

;;;>-----------------------------------------------------------------------------------------<
;;;         TN WAF
;;;>-----------------------------------------------------------------------------------------<

tn_ctrl = tnwaf(hgt_200_r_ctrl, u_200_clim_ctrl, v_200_clim_ctrl)
fx_ctrl = tn_ctrl[0]
fy_ctrl = tn_ctrl[1]

tn_sens = tnwaf(hgt_200_r_sens, u_200_clim_sens, v_200_clim_sens)
fx_sens = tn_sens[0]
fy_sens = tn_sens[1]

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
res@gsnStringFont = 22
; res@tmXBLabelFont = 7
; res@tmYLLabelFont = 7

res_sig = res
resh = res
res_tn = res
res_clm = res

resh@vpWidthF = 0.8
resh@vpHeightF = 0.4
resh@mpShapeMode = "FreeAspect"
resh@mpMinLatF = 10
resh@mpMaxLatF = 90
resh@mpMinLonF = 130
resh@mpMaxLonF = 310
resh@mpCenterLonF = 220
resh@mpFillOn = False
; res@mpLandFillColor = "white"
; res@mpFillDrawOrder = "PostDraw"
; res@mpOutlineDrawOrder = "PreDraw"
resh@mpGeophysicalLineColor = "gray40"
resh@mpGeophysicalLineThicknessF = 1.0
resh@gsnLeftStringOrthogonalPosF = 0.01
resh@gsnLeftStringFontHeightF = 28
resh@gsnRightStringOrthogonalPosF = 0.01
resh@gsnRightStringFontHeightF = 30

resh@tmXBMinorLengthF = 0
resh@tmYLMinorLengthF = 0
resh@tmXBMajorLengthF = 0.005
resh@tmYLMajorLengthF = 0.005
resh@tmXBLabelDeltaF = -0.5
resh@tmYLLabelDeltaF = -0.5
resh@tmXBLabelFontThicknessF = 1.5
resh@tmYLLabelFontThicknessF = 1.5
resh@tmXBLabelFontHeightF = 0.018
resh@tmYLLabelFontHeightF = 0.018
resh@tmXBTickSpacingF = 30
resh@tmYLTickSpacingF = 20
resh@tmXBMajorThicknessF = 1.5
resh@tmYLMajorThicknessF = 1.5
resh@tmXTOn = False
resh@tmYROn = False

resu = resh 
rest = resh


resh@cnLinesOn = False
resh@cnLineLabelsOn = False
resh@cnInfoLabelOn = False
resh@cnFillOn = True
resh@cnLevelSelectionMode = "ExplicitLevels"
; res@cnFillOpacityF = 0.5
; res1@cnFillPalette = "BlWhRe"
; res1@cnFillColors = (/30,33,36,39,42,-1,58,61,66,70,80/)
resh@cnFillPalette = "MPL_BrBG"
resh@cnLevels = (/-15,-12,-9,-6,-3,3,6,9,12,15/)
resh@cnFillColors = (/100,95,90,85,80,-1,45,40,35,30,25/)

resh@lbLabelBarOn = True
resh@lbOrientation = "Vertical"
resh@pmLabelBarWidthF = 0.04
resh@lbLabelFontHeightF = 0.015
resh@lbBoxEndCapStyle = "TriangleBothEnds"

resu@vpHeightF = 0.35
resu@mpMinLatF = 20
resu@mpMinLonF = 0
resu@mpMaxLonF = 360
resu@mpCenterLonF = 180
resu@tmXBTickSpacingF = 45
resu@cnFillOn = True
resu@cnLineLabelsOn = False
resu@cnLinesOn = False
resu@cnFillPalette = "BlWhRe"
resu@cnFillOpacityF = 0.8
resu@cnLevelSelectionMode = "ExplicitLevels"
resu@cnLevels = (/-2.5,-2,-1.5,-1,-0.5,0.5,1,1.5,2,2.5/)
resu@cnFillColors = (/33,35,37,39,41,-1,56,59,62,65,68/)

resu@lbLabelFontHeightF = 0.015
resu@lbLabelBarOn = True
resu@lbOrientation = "Vertical"
resu@pmLabelBarWidthF = 0.04
resu@lbBoxEndCapStyle = "TriangleBothEnds"


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
; res_sig@cnFillColors = (/1, 0/)
; res_sig@cnFillColors = (/"grey20", "Transparent"/)
res_sig@cnFillColors = (/"white", "Transparent"/)
res_sig@cnFillDotSizeF = 0.003
; res_sig@cnFillScaleF = 0.8
res_sig@cnFillDrawOrder = "PostDraw"


res_tn@vcGlyphStyle = "CurlyVector"
res_tn@vcMinDistanceF = 0.023
; res_tn@vcLineArrowThicknessF = 2.5
; res_tn@vcLineArrowThicknessF = 1.2
res_tn@vcLineArrowThicknessF = 0.8
res_tn@vcLineArrowColor = "black"
res_tn@vcLineArrowHeadMaxSizeF = 0.01
res_tn@vcLineArrowHeadMinSizeF = 0.01


res_tn@vcRefAnnoOn = True
res_tn@vcRefLengthF = 0.035
res_tn@vcRefAnnoBackgroundColor = "white"
res_tn@vcRefAnnoPerimOn = True   ;关闭边框
; res_tn@vcRefAnnoFont = 7
res_tn@vcRefAnnoFontHeightF = 0.008
res_tn@vcRefAnnoString1On =   False ;设置标注箭头上方不显示注释
res_tn@vcRefAnnoString2On = True   ;设置标注箭头下方显示注释
res_tn@vcRefAnnoFontThicknessF = 1.5

res_tn@vcRefAnnoOrthogonalPosF = -0.137
; res_tn@vcRefAnnoParallelPosF = 0.091
res_tn@vcRefAnnoParallelPosF = 0.08

; res_tn@vcRefAnnoOrthogonalPosF = -0.135
; res_tn@vcRefAnnoParallelPosF = 0.08
; res_tn@vcFillArrowFillColor = "orange2"
; res_tn@vcFillArrowEdgeColor = -1
res_tn@vcVectorDrawOrder = "PostDraw"


minlat = 65.
maxlat = 85.
minlon = -150.
maxlon = -60.

; rest@gsnLeftStringOrthogonalPosF = 0.3
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
rest@pmLabelBarWidthF = 0.04
rest@lbLabelFontHeightF = 0.015
rest@lbBoxEndCapStyle = "TriangleBothEnds"
rest@pmLabelBarOrthogonalPosF = 0.1

res_clm@cnLinesOn = True
res_clm@cnLineLabelsOn = True
res_clm@cnInfoLabelOn = False
res_clm@cnFillOn = False
res_clm@cnLineLabelFont = 7
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

wks = gsn_open_wks("x11", "./fig4-6")

;;;>-----------------------------------------------------------------------------------------<

gres = True
gres@gsLineColor = "green4"
gres@gsLineThicknessF = 2

lonw1 = 80
lone1 = 170
lats1 = 35
latn1 = 42
x1 = (/lonw1,lone1,lone1,lonw1,lonw1/)
y1 = (/lats1,lats1,latn1,latn1,lats1/)

resu@gsnLeftString = "(a) Reg. U250 onto AST in OBS."
plot_u200 = gsn_csm_contour_map(wks, u_200_r, resu)
plot_u200_sig = gsn_csm_contour(wks, confi_u200, res_sig)
overlay(plot_u200, plot_u200_sig)
plot_u200_clim = gsn_csm_contour(wks, u_200_clim, res_clm)
overlay(plot_u200, plot_u200_clim)
dum1 = gsn_add_polyline(wks, plot_u200_clim, x1, y1, gres)

resu@gsnLeftString = "(b) Diff. U250 (Exp_Ctrl - Exp_Arctic+2K)"
plot_u200_diff = gsn_csm_contour_map(wks, u_200_diff, resu)
plot_u200_diff_sig = gsn_csm_contour(wks, confi_u_200_diff, res_sig)
overlay(plot_u200_diff, plot_u200_diff_sig)
plot_u200_clim_model = gsn_csm_contour(wks, u_200_clim_sens, res_clm)
overlay(plot_u200_diff, plot_u200_clim_model)
dum2 = gsn_add_polyline(wks, plot_u200_clim_model, x1, y1, gres)

;;;>-----------------------------------------------------------------------------------------<

resh@gsnLeftString = "(c) Reg. Z250&WAF onto IPDC in Exp_Ctrl"
plot_h200_ctrl = gsn_csm_contour_map(wks, hgt_200_r_ctrl, resh)
plot_h200_ctrl_sig = gsn_csm_contour(wks, confi_h200_ctrl, res_sig)
overlay(plot_h200_ctrl, plot_h200_ctrl_sig)

resh@gsnLeftString = "(d) Reg. Z250&WAF onto IPDC in Exp_Arctic+2K"
plot_h200_sens = gsn_csm_contour_map(wks, hgt_200_r_sens, resh)
plot_h200_sens_sig = gsn_csm_contour(wks, confi_h200_sens, res_sig)
overlay(plot_h200_sens, plot_h200_sens_sig)

;;;>-----------------------------------------------------------------------------------------<

res_tn@vcRefMagnitudeF = 0.1   ;;; raw
res_tn@vcRefAnnoString2 = "0.1 m~S~2~N~/s~S~2~N~"
minf = 0.01     ;;  mask小值

fx_ctrl = where(sqrt((/fx_ctrl/)^2 + (/fy_ctrl/)^2) .gt. minf, fx_ctrl, 0)
fy_ctrl = where(sqrt((/fx_ctrl/)^2 + (/fy_ctrl/)^2) .gt. minf, fy_ctrl, 0)

fx_sens = where(sqrt((/fx_sens/)^2 + (/fy_sens/)^2) .gt. minf, fx_sens, 0)
fy_sens = where(sqrt((/fx_sens/)^2 + (/fy_sens/)^2) .gt. minf, fy_sens, 0)

plot_tn_ctrl = gsn_csm_vector(wks, fx_ctrl, fy_ctrl, res_tn)
overlay(plot_h200_ctrl, plot_tn_ctrl)

plot_tn_sens = gsn_csm_vector(wks, fx_sens, fy_sens, res_tn)
overlay(plot_h200_sens, plot_tn_sens)

;;;>-----------------------------------------------------------------------------------------<

lat_spacing = 5   ; use integers here
lon_spacing = 20
rest@cnLevels := (/-1,-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,1/)
rest@cnFillColors := (/33,36,39,42,45,-1,55,60,65,70,75/)

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

rest@gsnLeftString = "(e) Reg. T2m onto IPDC in Exp_Ctrl"
plot_t2m_ctrl = gsn_csm_contour_map(wks, t2m_r_ctrl, rest)
plot_sig_t2m_ctrl = gsn_csm_contour(wks, confi_t2m_ctrl, res_sig)
overlay(plot_t2m_ctrl, plot_sig_t2m_ctrl)
dum3 = gsn_add_polyline(wks, plot_sig_t2m_ctrl, x2, y2, gres)

add_lc_labels(wks, plot_sig_t2m_ctrl, minlat, maxlat, minlon, maxlon, lat_spacing, lon_spacing)

rest@gsnLeftString = "(f) Reg. T2m onto IPDC in Exp_Arctic+2K"
plot_t2m_sens = gsn_csm_contour_map(wks, t2m_r_sens, rest)
plot_sig_t2m_sens = gsn_csm_contour(wks, confi_t2m_sens, res_sig)
overlay(plot_t2m_sens, plot_sig_t2m_sens)
dum4 = gsn_add_polyline(wks, plot_sig_t2m_sens, x2, y2, gres)

add_lc_labels(wks, plot_sig_t2m_sens, minlat, maxlat, minlon, maxlon, lat_spacing, lon_spacing)

;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

pres = True
pres@gsnPanelXWhiteSpacePercent = 3
pres@gsnPanelYWhiteSpacePercent = 5
gsn_panel(wks, (/plot_u200, plot_u200_diff, \
                plot_h200_ctrl, plot_h200_sens, \
                plot_t2m_ctrl, plot_t2m_sens/), (/3, 2/), pres)


;;;>-----------------------------------------------------------------------------------------<
;;;>-----------------------------------------------------------------------------------------<

end
