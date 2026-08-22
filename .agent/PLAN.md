# 今案：ITEX XPU primitive 崩潰

- [x] 察新舊構建、驅動、依賴與旗標之異
- [x] 縮故於最小 LayerNorm/eltwise primitive，錄證
- [x] 列三策，擇一薦之，待允而改

次序：先查 build_changelog 與近提交；次驗環境及最小復現；末呈根因、三策、唯一薦策。

證：新構定 oneDNN 0e2a5bfe（v3.13），GPU runtime 為 SYCL；實載 Level-Zero GPU，OpenCL 僅 CPU；DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER 乃 undef。選 SYCL GPU 時，舊互操作路尋 OpenCL GPU，故 primitive 不成。

又證：最小 LayerNorm 與 GELU 皆敗於同一 Status 5；XLA 禁之亦然；舊 v3.11.3 wheel 於今 runtime 亦敗。故非模型形、Keras 或 XLA 之獨因，乃 GPU kernel 互操作契約失配。

今核：build_changelog 無近之 oneDNN 改；op_kernel.cc:526 自 2022 未變，乃 OP_REQUIRES_OK 共用報錯門，非根因。layer_norm_op.h:248、eltwise_base.h:175 皆捕 dnnl::error；其前 primitive_desc/primitive 生成乃觸發點。現載 libonednn_gpu_so.so 含 sycl_dev2ocl_dev 符號；生成設定 GPU=SYCL、DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER=OFF。故 Level-Zero 可見而 OpenCL GPU 不見時，失敗路徑已定。

又驗：令 OCL_ICD_FILENAMES 指向不存在之 Intel OpenCL DSO，ITEX 仍可載入、XPU 仍見 Level Zero；LayerNorm 仍於同一 OpenCL device not found / Status 5 敗。故 provider DSO 非初始化所需，然 legacy oneDNN primitive 仍硬需 OpenCL GPU。libonednn_gpu_so.so 新舊輪皆明列 libOpenCL.so.1；若 loader DSO 真不可解，當於載入前即敗。

三策既列：一啟 SYCL kernel compiler（薦）；二補 OpenCL GPU ICD；三令 LayerNorm/eltwise 避 oneDNN。未奉允，不改構建。

再驗：單跑 gelu_ln/XPU、ITEX_DISABLE_XLA=1、n=32、iters=1，仍於 primitive 建立敗；舊 OpenCL loader 回 CL_PLATFORM_NOT_FOUND_KHR，intelocl 僅報 CPU，故 GPU ICD 實無。

新證：以新 NEO 庫及 /etc/OpenCL/vendors/intel.icd 重跑全基準，XPU standard LayerNorm 成功，primitive failure 與 OpenCL device not found 皆消失；故實際修復為部署相容 NEO/OpenCL GPU provider，使既存 SYCL-to-OpenCL kernel 編譯路可行，非必改 oneDNN flag。

餘障：CPU JIT 報 resource 跨裝置；XPU JIT 報 ITEXLayerNorm 無 XLA_GPU_JIT kernel。二者皆非 primitive 建立故障；測腳應分別標注/避之。

別障：CPU JIT 所報 variable 跨裝置，乃測試腳本/Keras 資源置放之獨案；與 XPU standard 之 OpenCL device not found 分離，勿以 op_kernel 報告行混為一因。

# 舊案（保留）

- [x] 檢全檔、匿碼、動態載入、外聯
- [x] 察模型資產及字節碼
- [x] 驗運行依賴與網路行為
- [x] 判路徑、列證據、報風險

# 今案：核下一補丁之真偽與方略

- [x] 定案卷：`FINDINGS.md`、近提交、所謂 `PLAN.md`
- [x] 由計畫所稱失敗，循最短因果鏈核其實證
- [x] 審方案之責任層、契約、回歸面與更簡之策
- [x] 作守門結論：准、拒，或附條件准

今得根 `PLAN.md`。所稱因果：Keras2 LN 降為 FBN；匹配因 Const 與四維而失；遂落 oneDNN OpenCL-C。當核 dump、pattern、kernel 契約。

核定：失配與 OpenCL-C 退路俱真。然所據腳先以二維建 LN，後施四維；Keras 於 build 定 axis=1，遂正規化後三維，非末維。dump 之 gamma/beta `[1,128,1,1]` 即證。若剝 Reshape 而交末維 SYCL，諸維恰皆 128，形檢不敗而義變，乃默壞。

又：`isXlaAutoJitEnabled()` 但識全局 auto-JIT；真時初始化已令 remapper=0，假時顯式 `jit_compile=True` 仍可編譯。故局部守門既冗且不能護顯式 JIT。

再核數值：現退路明選 oneDNN `ocl:simple`；其 `simple.cl` 亦以 `E[x²]-E[x]²` 並截負值。故新路非始引此法；僅歸約次序異。無新舊輸出差證，不得以 Welford 為本補丁門檻。然驗證僅手工 log，且舊測明禁 constant folding，未鎖 Const 形。

守門：仍拒現案，然撤數值門檻。薦以語義守衛重擬：用正規四維建層；只納原位一維 gamma/beta，拒廣播 Reshape；驗 FBN NCHW、前後 Reshape、單位 Const/Fill、長度、epsilon；補正負數值及圖測。Welford 留獨案；毋加局部全局-XLA 分支。

復依域辭及 ADR 核之：XPU 絕不可選 OpenCL-C primitive，故不可薦補 ICD 或安於 FBN 退路；然以異義 LN 冒充原生 variant，亦違 device variant 契約。已於根 `PLAN.md` 加守門評注：准其方向，拒其 gamma 回溯；令先證正規末維圖，語義驗形，於上游守 XLA，且測為必需。CPU 三 ADR 與此 GPU 案分界，毋相援引。

用戶明加速為宗：少驗乃善，須辨時域。已改評注：毋增執行時驗證、適配、退路；remapper 於圖編譯時以最小 capability contract 擇核，核路純直。毋設廣測門檻；沿既有工作負載與圖證驗編譯、選核、去 OpenCL-C、留 epsilon、比一輸出。惟末維核不可承後綴維義，此非防禦驗證，乃防 miscompile 之編譯契約。
