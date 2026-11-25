# Consolidated TYROBP single-cell, CellChat and donor analyses.
# Native package functions and source styles are retained.
if(.Platform$OS.type=='windows') invisible(suppressWarnings(Sys.setlocale('LC_CTYPE','English_United States.utf8')))

# Annotation and quality-control sections, recovered 2026-11-25.
# All cached values originate from saved real Seurat objects or original plot tables.
# No count matrix, UMAP, clustering result, annotation or donor was simulated.
# Original compiled-App launchers are archived separately; do not source them as a batch.

.anno_need <- function(pkgs) {
  miss <- pkgs[!vapply(pkgs,requireNamespace,logical(1),quietly=TRUE)]
  if(length(miss))stop('Install the required R packages: ',paste(miss,collapse=', '))
}
.anno_input <- function(data_root,name) {
  candidates <- c(file.path(data_root,'annotation',name),file.path(data_root,'annotation_inputs',name),file.path(data_root,name))
  hit <- candidates[file.exists(candidates)]
  if(!length(hit))stop('Required annotation input missing: ',name,' under ',data_root)
  hit[1]
}
.anno_save <- function(draw,output_root,stem,width,height,dpi=300) {
  dir.create(output_root,recursive=TRUE,showWarnings=FALSE)
  .anno_need(c('ragg','svglite'))
  grDevices::cairo_pdf(file.path(output_root,paste0(stem,'.pdf')),width=width,height=height,family='Arial',bg='white')
  tryCatch(draw(),finally=grDevices::dev.off())
  ragg::agg_png(file.path(output_root,paste0(stem,'.png')),width=width,height=height,units='in',res=dpi,background='white')
  tryCatch(draw(),finally=grDevices::dev.off())
  svglite::svglite(file.path(output_root,paste0(stem,'.svg')),width=width,height=height,bg='white')
  tryCatch(draw(),finally=grDevices::dev.off())
  invisible(output_root)
}

# Native graphical objects produced by Seurat/ggplot2/clustree from the saved .qs.
# App UI style settings were not preserved for the broad plots/QC plots.
# These are native-function replays of real saved results, not an analysis rerun.
annotation_cached_native <- function(data_root,output_root,
    panels=c('Fig1A','Fig1D','Fig1E','Fig4A_afterQC','Fig4B','Fig4C','Fig4D')) {
  .anno_need(c('grid','ggraph','ggrepel','ragg','svglite'))
  for(panel in panels) {
    grob <- readRDS(.anno_input(data_root,paste0(panel,'_native_grob.rds')))
    dims <- if(panel=='Fig4A_afterQC')c(12,4) else if(panel=='Fig4D')c(9,8) else c(6,5)
    .anno_save(function(){grid::grid.newpage();grid::grid.draw(grob)},output_root,panel,dims[1],dims[2])
  }
  invisible(panels)
}

# Regenerate broad native plots from an external, actual Seurat .qs/.rds object.
# The bundled grobs can be replayed without transferring the large object.
annotation_from_seurat <- function(data_root,output_root,seurat_path,
    pre_qc_path=NULL,features=c('MMP9','PECAM1','CASP1','TYROBP'),
    group_features=c('PECAM1','CASP1','TYROBP'),reduction='umap',resolution_prefix='RNA_snn_res.') {
  .anno_need(c('Seurat','ggplot2','clustree','ggraph'))
  suppressPackageStartupMessages({library(Seurat);library(ggplot2);library(ggraph)})
  load_object <- function(p) {
    if(!file.exists(p))stop('External Seurat object not found: ',p)
    z <- if(grepl('\\.rds$',p,ignore.case=TRUE))readRDS(p) else { .anno_need('qs');qs::qread(p) }
    if(!inherits(z,'Seurat'))stop('Input must be an actual Seurat object: ',p)
    z
  }
  obj <- load_object(seurat_path)
  p <- list(
    Fig1A=DimPlot(obj,group.by='seurat_clusters',reduction=reduction,label=TRUE,pt.size=.1,raster=FALSE)+ggtitle(NULL),
    Fig1D=DotPlot(obj,features=features,group.by='cellType')+RotatedAxis(),
    Fig1E=DotPlot(obj,features=group_features,group.by='group')+RotatedAxis(),
    Fig4A_afterQC=VlnPlot(obj,features=c('nCount_RNA','nFeature_RNA','percent.mt','percent.rb'),ncol=4,pt.size=0),
    Fig4B=LabelPoints(VariableFeaturePlot(obj),points=head(VariableFeatures(obj),10),repel=TRUE),
    Fig4C=ElbowPlot(obj,ndims=30),
    Fig4D=clustree::clustree(obj[[]],prefix=resolution_prefix)
  )
  if(!is.null(pre_qc_path)) {
    before <- load_object(pre_qc_path)
    p$Fig4A_beforeQC <- VlnPlot(before,features=c('nCount_RNA','nFeature_RNA','percent.mt','percent.rb'),ncol=4,pt.size=0)
  }
  dir.create(output_root,recursive=TRUE,showWarnings=FALSE)
  for(n in names(p)) {
    dims <- if(grepl('QC',n))c(12,4)else c(6,5)
    ggsave(file.path(output_root,paste0(n,'.pdf')),p[[n]],width=dims[1],height=dims[2],device=cairo_pdf,bg='white')
    ggsave(file.path(output_root,paste0(n,'.png')),p[[n]],width=dims[1],height=dims[2],dpi=300,device=ragg::agg_png,bg='white')
  }
  invisible(p)
}

# Original circular-plot package. Coordinates use the original saved joint UMAP.
# The compiled App did not save its option values: defaults are exposed explicitly.
annotation_broad_circle <- function(data_root,output_root,
    contour.levels=c(.2,.3),pt.size=.5,bg.color='white',col.use=NULL,label.cex=.5) {
  .anno_need(c('plot1cell','circlize','dplyr','MASS','scales'))
  suppressPackageStartupMessages({library(plot1cell);library(dplyr);library(circlize)})
  d <- read.csv(.anno_input(data_root,'Broad_circle_plot1cell_coordinates.csv'),check.names=FALSE)
  # Package data include the true cell IDs. Preserve their saved factor order.
  d$Cluster <- factor(d$Cluster,levels=unique(d$Cluster))
  .anno_save(function(){
    circlize::circos.clear();on.exit(circlize::circos.clear())
    plot1cell::plot_circlize(d,do.label=TRUE,contour.levels=contour.levels,
      pt.size=pt.size,bg.color=bg.color,col.use=col.use,label.cex=label.cex)
  },output_root,'Fig1B_Broad_circular_UMAP',7,7)
}

# Broad-cell proportions are recalculated only from the archived real count table.
# Original compiled App body unavailable; this is a transparent ggalluvial renderer.
annotation_broad_alluvial <- function(data_root,output_root,palette=NULL) {
  .anno_need(c('ggplot2','ggalluvial'))
  suppressPackageStartupMessages({library(ggplot2);library(ggalluvial)})
  d <- read.csv(.anno_input(data_root,'Broad_cell_composition_counts.csv'))
  names(d) <- c('group','cellType','n')
  d$group <- factor(d$group,levels=c('Control','UC'))
  d$cellType <- factor(d$cellType,levels=c('B and Plasma cells','T cells','Stromal cells','Epithelium cells','Myeloid cells'))
  d$proportion <- d$n/ave(d$n,d$group,FUN=sum)
  if(is.null(palette))palette <- setNames(c('#E64B35','#4DBBD5','#00A087','#3C5488','#F39B7F'),levels(d$cellType))
  p <- ggplot(d,aes(x=group,stratum=cellType,alluvium=cellType,y=proportion,fill=cellType))+
    geom_flow(alpha=.35,colour=NA)+geom_stratum(width=.35,colour='#303030',linewidth=.25)+
    scale_fill_manual(values=palette,name='Cell Type')+theme_void(base_family='Arial')+
    theme(axis.text.x=element_text(angle=45,hjust=1,colour='#303030'),legend.position='right')
  .anno_save(function()print(p),output_root,'Fig1C_Broad_cell_alluvial',6,6)
  invisible(p)
}

annotation_myeloid_numeric <- function(data_root,output_root) {
 .anno_need(c('ggplot2','ggrepel','ragg','svglite'))
 suppressPackageStartupMessages({library(ggplot2);library(ggrepel)})
 out <- output_root
 for(sub in c('PDF','PNG','SVG','TIFF','tables','documentation'))dir.create(file.path(out,sub),recursive=TRUE,showWarnings=FALSE)
 d <- read.csv(.anno_input(data_root,'Myeloid1338_coordinates.csv'),check.names=FALSE)
 pal <- read.csv(.anno_input(data_root,'Myeloid_cluster_palette.csv'),colClasses='character')
 d$cluster <- factor(d$cluster,levels=pal$cluster)
 palette <- setNames(pal$colour,pal$cluster)
 x_limits <- range(d$UMAP_1)+c(-1,1)*diff(range(d$UMAP_1))*.065
 y_limits <- range(d$UMAP_2)+c(-1,1)*diff(range(d$UMAP_2))*.065
make_plot <- function(z) {
 # Medoid nearest the 2D median; labels are anchored to actual cells in each cluster.
 centers <- do.call(rbind,lapply(split(z,z$cluster),function(a){
  center <- apply(a[,c('UMAP_1','UMAP_2')],2,median)
  j <- which.min((a$UMAP_1-center[1])^2+(a$UMAP_2-center[2])^2)
  a[j,c('UMAP_1','UMAP_2','cluster')]
 }))
 set.seed(20261114)
 z<-z[sample.int(nrow(z)),]
 p <- ggplot(z,aes(UMAP_1,UMAP_2,colour=cluster))+
  geom_point(size=.85,alpha=.92,stroke=0)+
  geom_label_repel(data=centers,aes(label=cluster),seed=20261114,colour='#17212B',
   fill=scales::alpha('white',.83),size=3.4,fontface='bold',family='Arial',
   label.size=0,label.padding=grid::unit(.055,'lines'),box.padding=.22,
   point.padding=.12,min.segment.length=0,segment.colour='#686F74',segment.size=.28,
   max.overlaps=Inf,max.iter=10000,show.legend=FALSE)+
  scale_colour_manual(values=palette,drop=FALSE)+
  coord_equal(xlim=x_limits,ylim=y_limits,expand=FALSE)+
  labs(x='UMAP 1',y='UMAP 2',colour=NULL)+
  guides(colour=guide_legend(ncol=1,override.aes=list(size=2.4,alpha=1)))+
  theme_classic(base_family='Arial',base_size=11)+
  theme(axis.line=element_blank(),panel.border=element_rect(fill=NA,colour='#434B53',linewidth=.5),
    axis.ticks=element_line(colour='#434B53',linewidth=.35),axis.ticks.length=grid::unit(2.5,'pt'),
    axis.text=element_text(colour='#333B43',size=10),axis.title=element_text(size=11),
    axis.title.x=element_text(margin=margin(t=6)),axis.title.y=element_text(margin=margin(r=6)),
    legend.position='right',legend.text=element_text(size=10),legend.key.height=grid::unit(.34,'cm'),
    legend.key.width=grid::unit(.38,'cm'),legend.spacing.y=grid::unit(0,'pt'),
    legend.box.spacing=grid::unit(6,'pt'),plot.margin=margin(8,8,8,8))
 list(plot=p,centers=centers)
}
scopes <- list('01_Myeloid_clusters_before_annotation_1338'=d,
 '02_Myeloid_clusters_matching_main_1307'=d[d$included_in_main_display,])
plots<-list()
for (stem in names(scopes)) {
 result <- make_plot(scopes[[stem]]);p <- result$plot;plots[[stem]]<-p
 write.csv(result$centers,file.path(out,'tables',paste0(stem,'_label_anchors.csv')),row.names=FALSE)
 ggsave(file.path(out,'PDF',paste0(stem,'.pdf')),p,width=5.45,height=4.65,device=cairo_pdf,bg='white')
 ggsave(file.path(out,'PNG',paste0(stem,'.png')),p,width=5.45,height=4.65,dpi=600,device=ragg::agg_png,bg='white')
 ggsave(file.path(out,'SVG',paste0(stem,'.svg')),p,width=5.45,height=4.65,device=svglite::svglite,bg='white')
 ggsave(file.path(out,'TIFF',paste0(stem,'.tiff')),p,width=5.45,height=4.65,dpi=600,device=ragg::agg_tiff,compression='lzw',bg='white')
}
invisible(plots)

}

annotation_revised7 <- function(data_root,output_root) {
 .anno_need(c('ggplot2','circlize','MASS','scales','ragg','svglite'))
 suppressPackageStartupMessages(library(ggplot2))
 out <- output_root
 for(d in c('figures/PDF','figures/PNG','figures/SVG','plot_objects'))dir.create(file.path(out,d),recursive=TRUE,showWarnings=FALSE)
 palette <- read.csv(.anno_input(data_root,'Revised7_palette.csv'),check.names=FALSE)
 lv <- palette$cell_type;pal <- setNames(palette$color,lv);short <- setNames(palette$short_label,lv)
 dp <- read.csv(.anno_input(data_root,'Revised7_circle_coordinates.csv'),check.names=FALSE)
 ct <- read.csv(.anno_input(data_root,'Revised7_composition.csv'),check.names=FALSE)
 ct$display_group <- factor(ct$display_group,levels=lv);ct$group <- factor(ct$group,levels=c('Control','UC'))
 ring_plot <- function(){
  circlize::circos.clear();on.exit(circlize::circos.clear())
  par(mar=c(3.8,1,3.1,1),family='Arial',bg='white',xpd=NA)
  k <- length(lv)
  circlize::circos.par(start.degree=90,gap.degree=rep(3,k),cell.padding=c(0,0,0,0),track.margin=c(0,0),
   canvas.xlim=c(-1.21,1.21),canvas.ylim=c(-1.21,1.21),points.overflow.warning=FALSE)
  circlize::circos.initialize(factors=factor(lv,levels=lv),xlim=cbind(rep(0,k),rep(1,k)))
  circlize::circos.trackPlotRegion(ylim=c(0,1),track.height=.019,bg.border=NA,panel.fun=function(x,y){
   id <- circlize::get.cell.meta.data('sector.index');j <- match(id,lv)
   circlize::circos.rect(0,0,1,1,col=pal[id],border=NA)
   circlize::circos.text(.5,2.5,paste0(j,'. ',short[id]),facing='bending.outside',niceFacing=TRUE,cex=.68,col='#202020')
  })
  theta <- seq(0,2*pi,length.out=500);lines(.94*cos(theta),.94*sin(theta),col='#DADDE0',lwd=.7)
  den <- MASS::kde2d(dp$x,dp$y,n=250,lims=c(-.95,.95,-.95,.95))
  zz <- sort(as.numeric(den$z),decreasing=TRUE);cum <- cumsum(zz)/sum(zz)
  lev <- sort(unique(vapply(c(.65,.85),function(q)zz[which(cum>=q)[1]],numeric(1))))
  contour(den,levels=lev,add=TRUE,drawlabels=FALSE,col='#B6AB8F',lwd=.7)
  set.seed(20261113);ii <- sample.int(nrow(dp))
  points(dp$x[ii],dp$y[ii],pch=16,cex=.46,col=adjustcolor(pal[as.character(dp$display_group[ii])],alpha.f=.82))
  cen <- aggregate(cbind(x,y)~display_group,dp,median);cen <- cen[match(lv,cen$display_group),]
  lx <- cen$x+c(-.025,.03,.05,-.04,-.07,.035,.035)
  ly <- cen$y+c(.055,-.035,.03,-.055,.005,.055,-.04)
  segments(cen$x,cen$y,lx,ly,col='#666666',lwd=.6);points(lx,ly,pch=21,bg='white',col=pal,cex=1.35,lwd=1)
  text(lx,ly,labels=seq_along(lv),cex=.7,font=2,col='#202020')
  title(main='Revised myeloid annotation',cex.main=1.18,line=1.4)
  mtext('GSE214695 | Seven displayed populations | 1,307 cells',side=3,line=.1,cex=.77,col='#525A63')
  mtext('Mac-like denotes a putative tissue state with reference disagreement.',side=1,line=1.15,cex=.64,col='#626970')
  mtext('Original joint UMAP retained. Ring widths do not indicate abundance.',side=1,line=2.05,cex=.63,col='#626970')
 }
 save_base <- function(fun,stem,w,h){
  cairo_pdf(file.path(out,'figures/PDF',paste0(stem,'.pdf')),width=w,height=h,family='Arial');fun();dev.off()
  ragg::agg_png(file.path(out,'figures/PNG',paste0(stem,'.png')),width=w,height=h,units='in',res=600,background='white');fun();dev.off()
  svglite::svglite(file.path(out,'figures/SVG',paste0(stem,'.svg')),width=w,height=h,bg='white');fun();dev.off()
 }
 save_plot <- function(p,stem,w,h){
  ggsave(file.path(out,'figures/PDF',paste0(stem,'.pdf')),p,width=w,height=h,device=cairo_pdf,family='Arial',bg='white')
  ggsave(file.path(out,'figures/PNG',paste0(stem,'.png')),p,width=w,height=h,device=ragg::agg_png,dpi=600,bg='white')
  ggsave(file.path(out,'figures/SVG',paste0(stem,'.svg')),p,width=w,height=h,device=svglite::svglite,bg='white')
  saveRDS(p,file.path(out,'plot_objects',paste0(stem,'.rds')))
 }
 save_base(ring_plot,'01_Circular_UMAP_revised7',7.7,8.1)
 z <- ct[order(ct$group,ct$display_group),];z$ymax <- ave(z$proportion,z$group,FUN=cumsum);z$ymin <- z$ymax-z$proportion
 z$x <- ifelse(z$group=='Control',1,1.55)
 ribbons <- do.call(rbind,lapply(lv,function(g){
  a <- z[z$display_group==g&z$group=='Control',];b <- z[z$display_group==g&z$group=='UC',]
  tt <- seq(0,1,length.out=180);ss <- 3*tt^2-2*tt^3
  data.frame(x=1.125+.30*tt,ymin=a$ymin+(b$ymin-a$ymin)*ss,ymax=a$ymax+(b$ymax-a$ymax)*ss,display_group=factor(g,levels=lv))
 }))
 p <- ggplot()+geom_ribbon(data=ribbons,aes(x=x,ymin=ymin,ymax=ymax,fill=display_group,group=display_group),alpha=.62,color=NA)+
  geom_line(data=ribbons,aes(x=x,y=ymax,group=display_group),color='#737373',linewidth=.18,alpha=.65)+
  geom_rect(data=z,aes(xmin=x-.125,xmax=x+.125,ymin=ymin,ymax=ymax,fill=display_group),color='#42484E',linewidth=.32)+
  scale_fill_manual(values=pal,breaks=lv,labels=lv,name='Cell population')+
  scale_y_continuous(labels=scales::label_percent(accuracy=1),breaks=seq(0,1,.25),expand=expansion(mult=c(0,.01)))+
  scale_x_continuous(breaks=c(1,1.55),labels=c('Control','UC'),limits=c(.78,1.77),expand=expansion(mult=c(0,0)))+
  labs(title='Myeloid-cell composition',subtitle='GSE214695 | Revised annotation\nPooled cells: Control n = 390; UC n = 917',x=NULL,
   y='Proportion among displayed populations',caption='31 unresolved cells retained in the object but omitted from this display.\nRibbons connect categories, not cell trajectories; proportions are descriptive.')+
  theme_classic(base_family='Arial',base_size=12)+theme(plot.title=element_text(face='bold',size=16),plot.title.position='plot',
   plot.caption.position='plot',plot.subtitle=element_text(color='#505963',size=10.5,lineheight=1.15,margin=margin(b=15)),
   plot.caption=element_text(color='#505963',size=8.5,lineheight=1.2,hjust=0,margin=margin(t=14)),axis.text=element_text(color='#202020'),
   axis.title.y=element_text(size=11.5,margin=margin(r=8)),axis.line.x=element_blank(),axis.ticks.x=element_blank(),axis.text.x=element_text(size=12,face='bold'),
   legend.title=element_text(face='bold',size=11),legend.text=element_text(size=10),legend.position='right',
   legend.key.height=grid::unit(.63,'cm'),legend.key.width=grid::unit(.42,'cm'),plot.margin=margin(14,16,12,14))
 save_plot(p,'02_Alluvial_revised7',9.3,6.7)

}

annotation_tyrobp_hcl <- function(data_root,output_root) {
 .anno_need(c('ggplot2','colorspace','grid','ragg','svglite'))
 suppressPackageStartupMessages({library(ggplot2);library(colorspace);library(grid)})
 out <- output_root
 for(f in c('PDF','PNG','SVG','tables'))dir.create(file.path(out,f),recursive=TRUE,showWarnings=FALSE)
 d <- read.csv(.anno_input(data_root,'TYROBP_macrophage596_cells.csv'),check.names=FALSE)
types <- c('C1Q+ mature macrophages','ACOD1+ inflammatory macrophage-like cells','NRG1+ activated macrophage-like cells')
legend_names <- types
hues <- c(250,10,130)
stopifnot(nrow(d)==596,all(d$population%in%types),length(unique(d$donor))==12,
          sum(d$donor=='HC1')==31,sum(d$group=='Control')==167,
          all(d$expression>=0 & d$expression<=5),all(is.finite(d$x)),all(is.finite(d$y)))
map_colour <- function(value,population){
 t <- value/5
 lightness <- 86-66*t
 # Shared chroma bounded by the most restrictive hue at every value.
 maxc <- vapply(hues,function(h)colorspace::max_chroma(h,lightness),numeric(length(t)))
 chroma <- pmin(100*sqrt(t),.93*apply(maxc,1,min))
 result <- colorspace::hex(colorspace::polarLUV(lightness,chroma,hues[match(population,types)]),fixup=FALSE)
 stopifnot(!anyNA(result))
 result
}
d$display_colour <- map_colour(d$expression,d$population)
steps <- seq(0,5,length.out=501)
pal <- do.call(rbind,lapply(seq_along(types),function(i){
 rgbhex <- map_colour(steps,rep(types[i],length(steps)))
 recovered <- colorspace::coords(as(colorspace::hex2RGB(rgbhex),'polarLUV'))[,'L']
 data.frame(population=types[i],expression=steps,colour=rgbhex,target_L=86-66*steps/5,actual_L=recovered)
}))
stopifnot(max(abs(pal$target_L-pal$actual_L))<.6)
lum_by_value <- aggregate(actual_L~expression,pal,function(v)diff(range(v)))
stopifnot(max(lum_by_value$actual_L)<.6)
write.csv(d,file.path(out,'tables/Plotted_cells_with_display_colours.csv'),row.names=FALSE)
write.csv(pal,file.path(out,'tables/Shared_expression_colour_key.csv'),row.names=FALSE)
write.csv(as.data.frame(table(factor(d$population,levels=types),d$group)),file.path(out,'tables/Cell_counts.csv'),row.names=FALSE)
label_ink <- setNames(map_colour(rep(4.6,3),types),types)
draw_panel <- function(condition){
 z <- if(condition=='Overall')d else d[d$group==condition,]
 z <- z[order(z$expression),]
 labs <- data.frame(population=types,
    label=c('C1Q+ mature macrophages','ACOD1+ macrophage-like','NRG1+ macrophage-like'),
    x=c(-2.1,-1.45,2.65),y=c(4.78,-6.08,-1.20))
 labs <- labs[labs$population%in%z$population,]
 p <- ggplot(z,aes(x,y))+
  geom_point(aes(colour=display_colour),size=.86,stroke=0,alpha=1)+scale_colour_identity()+
  scale_x_continuous(expand=expansion(mult=0))+scale_y_continuous(expand=expansion(mult=0))+
  coord_fixed(xlim=c(-6.5,5.2),ylim=c(-6.5,5.2),expand=FALSE,clip='off')+
  labs(title=bquote(italic(TYROBP)~'|'~.(condition)),x='UMAP 1',y='UMAP 2')+
  theme_classic(base_family='Arial',base_size=8)+
  theme(plot.title=element_text(size=10,hjust=0,color='#161616',margin=margin(b=4)),
    plot.title.position='panel',plot.margin=margin(6,4,5,5),
    axis.title=element_text(size=8,color='#232323'),
    axis.title.x=element_text(margin=margin(t=3)),axis.title.y=element_text(margin=margin(r=3)),
    panel.border=element_rect(fill=NA,color='#353535',linewidth=.3),
    axis.line=element_blank(),axis.ticks=element_blank(),axis.text=element_blank())
 centers <- aggregate(cbind(x,y)~population,z,median)
 nrg <- centers[centers$population==types[3],]
 if(nrow(nrg))p <- p+geom_segment(data=nrg,aes(x=.55,y=-1.2,xend=x+.15,yend=y),
       inherit.aes=FALSE,linewidth=.22,color=label_ink[types[3]])
 for(i in seq_len(nrow(labs)))p <- p+geom_label(data=labs[i,,drop=FALSE],
    aes(x=x,y=y,label=label),inherit.aes=FALSE,size=2.15,fontface='bold',family='Arial',
    color=label_ink[labs$population[i]],fill='white',linewidth=0,label.padding=unit(.08,'lines'))
 grid.newpage();pushViewport(viewport(x=1.8/5.45,y=.5,width=3.6/5.45,height=1))
 grid.draw(ggplotGrob(p));popViewport()
 # Equal ticks, dimensions, direction and numerical range for every subtype bar.
 grid.text('TYROBP expression (log-normalized)',x=3.48/5.45,y=.89,just='left',gp=gpar(fontfamily='Arial',fontsize=6.9,col='#252C32'))

 present <- which(types%in%z$population)
 ys <- if(length(present)==3)c(.66,.43,.20) else c(.62,.35)
 for(k in seq_along(present)){
   i <- present[k];y <- ys[k]
   grid.text(legend_names[i],x=3.48/5.45,y=y+.050,just='left',gp=gpar(fontfamily='Arial',fontsize=6.1,fontface='bold',lineheight=1.08,col=label_ink[types[i]]))
   grad <- matrix(map_colour(steps,rep(types[i],length(steps))),nrow=1)
   grid.raster(as.raster(grad),x=4.37/5.45,y=y,width=1.78/5.45,height=.029,interpolate=FALSE)
   for(v in c(0,2.5,5))grid.text(format(v,trim=TRUE),x=3.48/5.45+(1.78/5.45)*v/5,y=y-.044,
           gp=gpar(fontfamily='Arial',fontsize=7,col='#48515B'))
 }
}
for(condition in c('Overall','Control','UC')){
 stem <- paste0('TYROBP_Subtype_Expression_',condition)
 cairo_pdf(file.path(out,'PDF',paste0(stem,'.pdf')),width=5.45,height=3.55,family='Arial',bg='white')
 draw_panel(condition);dev.off()
 ragg::agg_png(file.path(out,'PNG',paste0(stem,'.png')),width=5.45,height=3.55,units='in',res=600,background='white')
 draw_panel(condition);dev.off()
 svglite::svglite(file.path(out,'SVG',paste0(stem,'.svg')),width=5.45,height=3.55,bg='white')
 draw_panel(condition);dev.off()
}

}

# Explicit interactive entry point for the original compiled App stages.
# This function never installs packages or sources the complete historical launcher.
# Configure the licence/token in your own R environment if the installed package needs it.
annotation_original_app <- function(data_root,output_root,
    app=c('runSeurat1App','runSeurat3App','runSeurat4App','runSeurat5App','runSeurat6App',
          'runSeurat7App','runSeurat9App','runSeurat10_4App','runSeurat16App'),
    seurat_path=NULL) {
  app <- match.arg(app);.anno_need('Vcgxr410scRNA')
  if(!is.null(seurat_path)) {
    .anno_need('qs');obj <- if(grepl('\\.rds$',seurat_path,ignore.case=TRUE))readRDS(seurat_path)else qs::qread(seurat_path)
    if(!inherits(obj,'Seurat'))stop('Expected original Seurat object')
    assign('seurat',obj,envir=.GlobalEnv)
  }
  dir.create(output_root,recursive=TRUE,showWarnings=FALSE)
  old <- getwd();on.exit(setwd(old));setwd(output_root)
  getFromNamespace(app,'Vcgxr410scRNA')()
}

ANNO_STAGES <- list(
 annotation_native=list(panels=c('1A','1D','1E','4A_afterQC','4B','4C','4D'),
   fun=annotation_cached_native,mode='cached_native_functions',default=TRUE),
 annotation_circle=list(panels='1B',fun=annotation_broad_circle,
   mode='native_package_with_saved_coordinates_App_settings_unavailable',default=TRUE),
 annotation_alluvial=list(panels='1C',fun=annotation_broad_alluvial,
   mode='real_count_table_native_ggalluvial_App_body_unavailable',default=TRUE),
 annotation_myeloid=list(panels='1F',fun=annotation_myeloid_numeric,
   mode='original_native_code_and_saved_coordinates',default=TRUE),
 annotation_revised7=list(panels=c('1G','1H'),fun=annotation_revised7,
   mode='original_native_code_and_saved_tables',default=TRUE),
 annotation_tyrobp=list(panels=c('1I','1J'),fun=annotation_tyrobp_hcl,
   mode='original_native_code_and_saved_expression_coordinates',default=TRUE),
 annotation_external_seurat=list(panels=c('1A','1D','1E','4A','4B','4C','4D'),fun=annotation_from_seurat,
   mode='external_Seurat_object_required_beforeQC_unavailable',default=FALSE),
 annotation_app=list(panels=c('1B','1C','4A'),fun=annotation_original_app,
   mode='interactive_original_compiled_App_external_objects_and_UI_settings',default=FALSE)
)

run_annotation_stage <- function(stage,data_root,output_root,...) {
  if(!stage%in%names(ANNO_STAGES))stop('Unknown annotation stage: ',stage)
  ANNO_STAGES[[stage]]$fun(data_root=data_root,output_root=output_root,...)
}


# Attachment 2: original-strength CellChat A-E and Slingshot trajectory F-H.
# Recovered original R plotting functions, with portable input/output paths.
# This stage reads fitted model objects; it does not refit CellChat from raw cells.
# Original full analysis scripts and file hashes are in cellchat_originals/ and
# cellchat_manifest.json. No synthetic values or screenshot-derived coordinates.

.cellchat_find_data <- function(data_root) {
  candidates <- unique(c(file.path(data_root,"fig02_cellchat"),
    file.path(data_root,"cellchat","fig02_cellchat"),
    file.path(data_root,"cellchat_inputs","fig02_cellchat"),data_root))
  valid <- vapply(candidates, function(x) file.exists(file.path(x,"input","full_00.rds")), logical(1))
  if (!any(valid)) stop("CellChat input missing: input/full_00.rds under fig02_cellchat. See cellchat_manifest.json.")
  normalizePath(candidates[which(valid)[1]], winslash="/", mustWork=TRUE)
}

run_fig02_cellchat_trajectory <- function(data_root, output_root, ...) {
  required <- c("CellChat","ComplexHeatmap","ggplot2","grid","slingshot","uwot",
                "cowplot","pdftools","RColorBrewer","circlize")
  absent <- required[!vapply(required,requireNamespace,logical(1),quietly=TRUE)]
  if (length(absent)) stop("Install required R packages: ",paste(absent,collapse=", "))
  if (.Platform$OS.type=="windows" && !l10n_info()[["UTF-8"]]) {
    invisible(Sys.setlocale("LC_CTYPE","English_United States.utf8"))
  }
  input_dir <- .cellchat_find_data(data_root)
  result_dir <- file.path(output_root,"fig02_cellchat_trajectory")
  dir.create(result_dir, recursive=TRUE, showWarnings=FALSE)
  result_dir <- normalizePath(result_dir,winslash="/",mustWork=TRUE)
  stage <- new.env(parent=environment())
  stage$root <- input_dir
  stage$project_dir <- result_dir
  stage$f <- local({ input <- input_dir; function(...) file.path(input,...) })
  stage$plot_environment <- stage
  stage$.cellchat_trajectory_source <- .CELLCHAT_TRAJECTORY_SOURCE
  eval(parse(text=.CELLCHAT_NATIVE_OBJECTS,encoding="UTF-8"),envir=stage)
  eval(parse(text=.CELLCHAT_ORIGINAL_LAYOUT,encoding="UTF-8"),envir=stage)
  audit_dir <- file.path(result_dir,"audit_tables")
  dir.create(audit_dir,showWarnings=FALSE)
  write.csv(stage$ab,file.path(audit_dir,"A_B_directed_counts_and_strength.csv"),row.names=FALSE)
  write.csv(stage$dots,file.path(audit_dir,"C_native_CellChat_bubble_data.csv"),row.names=FALSE)
  write.csv(stage$outgoing,file.path(audit_dir,"D_outgoing_pathway_strength.csv"))
  write.csv(stage$incoming,file.path(audit_dir,"E_incoming_pathway_strength.csv"))
  write.csv(stage$plot_df,file.path(audit_dir,"F_G_H_trajectory_cells.csv"),row.names=FALSE)
  write.csv(stage$donors,file.path(audit_dir,"H_TYROBP_within_donor_correlations.csv"),row.names=FALSE)
  invisible(list(stage="fig02_cellchat_trajectory",output_dir=result_dir,
    cells_cellchat=length(stage$cc@idents), cells_trajectory=nrow(stage$plot_df),
    donors_trajectory=length(unique(stage$plot_df$donor)),
    focal_outgoing_count=sum(stage$ab$count[stage$ab$source==stage$focal]),
    focal_incoming_count=sum(stage$ab$count[stage$ab$target==stage$focal]),
    bubble_predictions=nrow(stage$dots), ligand_receptor_pairs=length(unique(stage$dots$interaction_name)),
    pathways=length(stage$pw), within_donor_sign_test_p=stage$p_sign,
    pdf=file.path(result_dir,"Figure7_original_strength.pdf"),
    png=file.path(result_dir,"Figure7_original_strength_600dpi.png")))
}

CELLCHAT_STAGES <- list(fig02_cellchat_trajectory=run_fig02_cellchat_trajectory)


.CELLCHAT_NATIVE_OBJECTS <- paste0(
  "# Verified CellChat and Slingshot inputs and original R plotting functions for Figure 7.\n# This file is sourced by Figure7_relayout.R, which defines input paths and the plotting environment.\nsuppressPackageStartupMessages({\n  library(CellChat)\n  library(ComplexHeatmap)\n  library(ggplot2)\n  library(grid)\n  library(slingshot)\n  library(uwot)\n})\n# No package theme or palette is replaced. Preserve the relative scaling and\n# cell abundances of the original CellChat object when restricting the network.\ncc <- readRDS(f(\"input\", \"full_00.rds\"))\ndonor_counts <- read.csv(f(\"input\",\"Supplementary_Table_6_UC_donor_contributions.csv\"),check.names=FALSE)\nraw_levels <- levels(cc@idents)\n# netVisual_circle fixes its font to Helvetica. The ASCII plus remains legible\n# on systems where that font has no superscript plus glyph.\nlabels <- c(ACOD1=\"ACOD1+\", C1Q=\"C1Q+\", DC=\"DC/DC-like\",\n            FCN1=\"FCN1+\", Mast=\"Mast\", NRG1=\"NRG1+\", Neutrophils=\"Neutrophils\")\nstopifnot(setequal(raw_levels, names(labels)), length(cc@idents) == 917, nrow(donor_counts) == 6)\nstopifnot(all(as.integer(table(cc@idents)) == colSums(donor_counts[,raw_levels])))\ngroup_size <- setNames(as.numeric(table(cc@idents)), unname(labels[raw_levels]))\nfocal <- unname(labels[\"ACOD1\"])\npartners <- setdiff(names(group_size), focal)\noriginal_count <- cc@net$count\noriginal_weight <- cc@net$weight\n# Set the presentation labels, preserving the tested group order in each dimension.\ncc@meta$display <- factor(labels[as.character(cc@idents)], levels = unname(labels[raw_levels]))\ncc <- setIdent(cc, ident.use=\"display\", display.warning=FA",
  "LSE)\ncc <- aggregateNet(cc, thresh=.05)\nstopifnot(identical(unname(cc@net$count),unname(original_count)),\n          isTRUE(all.equal(unname(cc@net$weight),unname(original_weight))))\ncount <- cc@net$count\nstrength <- cc@net$weight\nbetween <- outer(rownames(count) == focal, colnames(count) == focal, FUN = xor)\nab <- expand.grid(source=rownames(count),target=colnames(count), stringsAsFactors=FALSE)\nab$count <- as.vector(count)\nab$strength <- as.vector(strength)\nab <- ab[as.vector(between),]\nstopifnot(nrow(ab) == 12L, sum(ab$count[ab$source==focal]) == 102L,\n          sum(ab$count[ab$target==focal]) == 112L)\ncc <- computeCommunProbPathway(cc, thresh=.05-1e-12)\ncc <- netAnalysis_computeCentrality(cc, slot.name=\"netP\")\npw <- cc@netP$pathways\nstopifnot(length(pw) == 31L)\nnetp <- cc@netP$prob\noutgoing <- vapply(cc@netP$centr, function(x) unname(x$outdeg), numeric(length(group_size)))\nincoming <- vapply(cc@netP$centr, function(x) unname(x$indeg), numeric(length(group_size)))\nrownames(outgoing) <- rownames(incoming) <- levels(cc@idents)\nstopifnot(isTRUE(all.equal(unname(outgoing),unname(apply(netp,c(1,3),sum)),tolerance=1e-9)),\n          isTRUE(all.equal(unname(incoming),unname(apply(netp,c(2,3),sum)),tolerance=1e-9)))\n\n# CellChat rounds edge labels to one decimal by default. Use three decimals\n# for unscaled strengths in a local copy of the native plotting function.\nstrength_circle <- CellChat::netVisual_circle\nenvironment(strength_circle) <- list2env(\n  list(round = function(x, digits) base::round(x, digits = 3L)),\n  parent = environment(strength_circle)\n)\n\n# Sections 13 and 14 of ",
  "cellchat-\u65b0.R use netVisual_circle() on rows or\n# columns. Keep outgoing and incoming edges in their respective circles.\nnetwork <- function(metric, direction) {\n  is_count <- metric==\"count\"\n  full <- if (is_count) count else strength\n  draw_circle <- if (is_count) CellChat::netVisual_circle else strength_circle\n  mat <- full\n  diag(mat) <- 0\n  sources <- if (direction==\"out\") focal else NULL\n  targets <- if (direction==\"in\") focal else NULL\n  heading <- paste(if (direction==\"out\") \"Outgoing\" else \"Incoming\",\n                   if (is_count) \"interaction count\" else \"strength\")\n  draw_circle(mat, sources.use=sources, targets.use=targets,\n              vertex.weight=group_size, weight.scale=TRUE,\n              edge.weight.max=max(full), label.edge=TRUE,\n              vertex.label.cex=.82, edge.label.cex=if (is_count) .75 else .65,\n              margin=.45, title.name=heading, text.y=1.43)\n}\n\n# Section 15 of cellchat-\u65b0.R uses netVisual_bubble with sources.use/targets.use\n# and angle.x=45. Here the two directions appear in one native bubble plot.\n# Only the plot copy is masked. No probability or P value for a focal edge is changed.\nfocal_cc <- cc\nfor (i in seq_len(nrow(count))) {\n  for (j in seq_len(ncol(count))) {\n    if (!between[i,j]) {\n      focal_cc@net$prob[i,j,] <- 0\n      focal_cc@net$pval[i,j,] <- 1\n    }\n  }\n}\nbubble_data <- netVisual_bubble(focal_cc, sources.use=c(focal,partners),\n                                targets.use=c(partners,focal), remove.isolate=TRUE,\n                                angle.x=45, return.data=TRUE)\np_bubble <- bubble_data$gg.obj\n# Use a PDF",
  " gradient for the colour key. These are the title and width used\n# by netVisual_bubble; the data points, scales and labels come from CellChat.\np_bubble <- p_bubble + guides(colour=guide_colourbar(title=\"Commun. Prob.\",\n                                                     barwidth=0.5, display=\"gradient\"))\ndots <- bubble_data$communication\nexpected <- subsetCommunication(cc,thresh=.05)\nexpected <- expected[xor(expected$source==focal,expected$target==focal),]\nkey <- function(x) paste(x$source,x$target,x$interaction_name,sep=\"|\")\nind <- match(key(dots),key(expected))\nstopifnot(nrow(dots)==214L, length(unique(dots$interaction_name))==62L,\n          length(unique(dots$source.target))==12L, !anyNA(ind), !anyDuplicated(ind),\n          max(abs(dots$prob.original-expected$prob[ind]))<1e-12)\n# Section 17. CellChat adds its own marginal bars, group colours and relative\n# strength legend. Use all pathways; sizes are physical layout parameters.\nrole_heatmap <- function(pattern, width_cm=4.6, height_cm=10.9, font_size=8.4) {\n  h <- netAnalysis_signalingRole_heatmap(cc, pattern=pattern, signaling=pw,\n                                         width=width_cm, height=height_cm, font.size=font_size)\n  # If there are exactly 100 observed values, ComplexHeatmap 2.26 may treat\n  # CellChat's 100-colour continuous BuGn scale as a discrete colour key.\n  # Preserve the identical native colours and numeric range, explicitly\n  # specifying interpolation so that the rounded legend ticks can be drawn.\n  if (is.numeric(h@matrix) && h@matrix_color_mapping@type==\"discrete\") {\n    native_palette <- grDevice",
  "s::colorRampPalette(RColorBrewer::brewer.pal(9,\"BuGn\"))(100)\n    bounds <- range(h@matrix, na.rm=TRUE)\n    colfun <- circlize::colorRamp2(seq(bounds[1],bounds[2],length.out=100), native_palette)\n    h@matrix_color_mapping <- ComplexHeatmap::ColorMapping(name=\"Relative strength\",\n                                                           col_fun=colfun,na_col=\"white\")\n  }\n  h\n}\n\n# Retrieve the verified revised Slingshot fit, not the legacy M1/M2 model.\nfit <- readRDS(f(\"input\",\"slingshot_primary.rds\"))\npc <- slingReducedDim(fit)\ncell <- read.csv(f(\"input\",\"trajectory_primary_cells.csv\"),stringsAsFactors=FALSE)\npt <- drop(slingPseudotime(fit))\ngenes <- readLines(f(\"input\",\"trajectory_ordering_genes.txt\"))\nstopifnot(nrow(cell)==568L, length(unique(cell$donor))==6L, !anyDuplicated(cell$cell),\n          ncol(pc)==10L, all(abs(cell$PC1-pc[,1]) < 1e-9), all(abs(cell$PC2-pc[,2]) < 1e-9),\n          length(genes)==1500L, !any(grepl(\"^TYROBP([.][0-9]+)?$\",genes,ignore.case=TRUE)),\n          as.character(slingLineages(fit)[[1]][1])==\"FCN1\",\n          max(abs(pt/max(pt)-cell$time))<1e-9)\n# The fitted object uses successive Cell-1, Cell-2, ... internal IDs. The\n# archived barcode table contains the very same unique PC1/PC2 coordinates\n# and pseudotime in the same order, as checked above.\nstopifnot(!anyDuplicated(as.data.frame(pc[,1:2])))\nif (!is.null(rownames(pc))) stopifnot(identical(rownames(pc),cell$cell) ||\n                                    identical(rownames(pc),paste0(\"Cell-\",seq_len(nrow(pc)))))\nset.seed(71021)\nxy <- uwot::umap(pc, n_neighbors=30, n_components=2, min_dist=.35, m",
  "etric=\"euclidean\",\n                 init=\"spectral\", spread=1, n_threads=1, n_sgd_threads=1, verbose=FALSE)\nplot_df <- data.frame(cell=cell$cell, donor=cell$donor, UMAP_1=xy[,1], UMAP_2=xy[,2],\n                      cellType=factor(labels[cell$state],levels=unname(labels[c(\"FCN1\",\"ACOD1\",\"NRG1\",\"C1Q\")])),\n                      pseudotime=cell$time, TYROBP_expr=cell$TYROBP)\nplot_df$donor_weight <- 1/ave(rep(1,nrow(plot_df)), plot_df$donor, FUN=sum)\n# Use exactly the same cell-group colours as the native CellChat plots.\npalette_cc <- setNames(CellChat::scPalette(length(group_size)),names(group_size))\ncelltype_cols <- palette_cc[levels(plot_df$cellType)]\n\n# The original ggplot2 calls for p1/p2/p3 are evaluated directly from the\n# archived script. Only presentation identity, root/values and the statistically\n# justified changes to the descriptive smoother are specified below.\ntrajectory_source <- parse(text=.cellchat_trajectory_source, encoding=\"UTF-8\")\nget_source_plot <- function(symbol) {\n  matched <- vapply(trajectory_source, function(x)\n    is.call(x) && identical(x[[1]],as.name(\"<-\")) &&\n      identical(x[[2]],as.name(symbol)), logical(1))\n  stopifnot(sum(matched)==1L)\n  eval(trajectory_source[[which(matched)]][[3]], envir=plot_environment)\n}\np_states <- get_source_plot(\"p1\") + labs(title=\"Cell subsets\")\np_time <- get_source_plot(\"p2\")\n# Preserve the original colour scale with a vector gradient in the PDF legend.\np_time <- p_time + guides(colour=guide_colourbar(display=\"gradient\"))\np_expression <- get_source_plot(\"p3\")\n# Preserve the original LOESS geometry and black line.",
  " Equal donor weights\n# avoid allowing a large donor to dominate. A cell-independent SE band is\n# inappropriate here, so inference is reported separately at donor level.\nsmooth_id <- which(vapply(p_expression$layers,function(z) inherits(z$stat,\"StatSmooth\"),logical(1)))\nstopifnot(length(smooth_id)==1L)\np_expression$layers[[smooth_id]]$mapping <- aes(weight=donor_weight)\np_expression$layers[[smooth_id]]$stat_params$se <- FALSE\ndonors <- do.call(rbind, lapply(split(plot_df,plot_df$donor), function(dd) {\n  data.frame(donor=dd$donor[1],cells=nrow(dd),rho=cor(dd$pseudotime,dd$TYROBP_expr,method=\"spearman\"))\n}))\nold_donors <- read.csv(f(\"input\",\"trajectory_donor_correlations.csv\"))\nold_donors <- old_donors[old_donors$fit==\"primary\",]\nstopifnot(isTRUE(all.equal(donors$rho,old_donors$rho[match(donors$donor,old_donors$donor)],tolerance=1e-10)))\np_sign <- binom.test(sum(donors$rho>0),sum(donors$rho!=0),p=.5)$p.value\nstopifnot(isTRUE(all.equal(p_sign,.6875,tolerance=1e-12)))\n"
)

.CELLCHAT_TRAJECTORY_SOURCE <- paste0(
  "############################################################\n## UC \u5de8\u566c\u7ec6\u80de\u62df\u65f6\u5e8f\u5206\u6790\uff08\u4ec5\u4fdd\u7559 M1 + M2\uff09\n## \u76ee\u7684\uff1a\u89c2\u5bdf TYROBP \u662f\u5426\u6cbf M2 -> M1 \u8f68\u8ff9\u5347\u9ad8\n############################################################\n\n############################################################\n## 0. \u52a0\u8f7d R \u5305\n############################################################\n\npackages <- c(\n  \"Seurat\", \"monocle3\", \"dplyr\", \"ggplot2\",\n  \"patchwork\", \"Matrix\", \"tibble\", \"ggpubr\", \"viridis\"\n)\n\nfor (pkg in packages) {\n  if (!requireNamespace(pkg, quietly = TRUE)) {\n    stop(paste0(\"\u7f3a\u5c11 R \u5305\uff1a\", pkg, \"\uff0c\u8bf7\u5148\u5b89\u88c5\u540e\u518d\u8fd0\u884c\u3002\"))\n  }\n}\n\nlibrary(Seurat)\nlibrary(monocle3)\nlibrary(dplyr)\nlibrary(ggplot2)\nlibrary(patchwork)\nlibrary(Matrix)\nlibrary(tibble)\nlibrary(ggpubr)\nlibrary(viridis)\n\nset.seed(1234)\n\n############################################################\n## 1. \u8f93\u51fa\u76ee\u5f55\n############################################################\n\noutdir <- \"TYROBP_M1_M2_pseudotime\"\ndir.create(outdir, showWarnings = FALSE)\n\n############################################################\n## 2. \u53ea\u4fdd\u7559 M1 \u548c M2 macrophages\n############################################################\n\nseurat$cellType <- as.character(seurat$cellType)\n\nmacro_types <- c(\n  \"M1 macrophages\",\n  \"M2 macrophages\"\n)\n\nmac <- subset(\n  seurat,\n  subset = cellType %in% macro_types\n)\n\ncat(\"\u63d0\u53d6\u540e\u7684\u7ec6\u80de\u6570\uff1a\\n\")\nprint(table(mac$cellType))\n\n############################################################\n## 3. \u9009\u62e9 assay\n############################################################\n\nassay_names <- names(mac@assays)\nprint(assay_names)\n\nif (\"RNA\" %in% assay_names) {\n  DefaultAssay(mac) <- \"RNA\"\n} else if (\"SCT\" %in% assay_name",
  "s) {\n  DefaultAssay(mac) <- \"SCT\"\n} else {\n  DefaultAssay(mac) <- assay_names[1]\n}\n\ncat(\"\u5f53\u524d DefaultAssay\uff1a\", DefaultAssay(mac), \"\\n\")\n\n############################################################\n## 4. \u91cd\u65b0\u6807\u51c6\u5316\u3001\u964d\u7ef4\u3001\u805a\u7c7b\n############################################################\n\nif (DefaultAssay(mac) == \"RNA\") {\n  mac <- NormalizeData(mac, normalization.method = \"LogNormalize\", scale.factor = 10000)\n  mac <- FindVariableFeatures(mac, selection.method = \"vst\", nfeatures = 2000)\n  mac <- ScaleData(mac, features = rownames(mac))\n  mac <- RunPCA(mac, features = VariableFeatures(mac), npcs = 30)\n} else {\n  mac <- ScaleData(mac, features = rownames(mac))\n  mac <- RunPCA(mac, features = VariableFeatures(mac), npcs = 30)\n}\n\nmac <- FindNeighbors(mac, dims = 1:20)\nmac <- FindClusters(mac, resolution = 0.3)\nmac <- RunUMAP(mac, dims = 1:20)\n\n############################################################\n## 5. \u63d0\u53d6 TYROBP \u8868\u8fbe\u5e76\u5206\u7ec4\n############################################################\n\nif (!\"TYROBP\" %in% rownames(mac)) {\n  stop(\"\u5f53\u524d\u5bf9\u8c61\u4e2d\u6ca1\u6709 TYROBP\uff0c\u8bf7\u68c0\u67e5\u57fa\u56e0\u540d\u3002\")\n}\n\ntyrobp_expr <- FetchData(mac, vars = \"TYROBP\")[, 1]\nmac$TYROBP_expr <- tyrobp_expr\n\n# \u5728\u5168\u90e8\u5de8\u566c\u7ec6\u80de\u4e2d\u6309\u4e2d\u4f4d\u6570\u5206\u7ec4\ncutoff_tyrobp <- median(mac$TYROBP_expr, na.rm = TRUE)\n\nmac$TYROBP_group <- ifelse(\n  mac$TYROBP_expr > cutoff_tyrobp,\n  \"TYROBP_high\",\n  \"TYROBP_low\"\n)\n\nmac$TYROBP_group <- factor(\n  mac$TYROBP_group,\n  levels = c(\"TYROBP_low\", \"TYROBP_high\")\n)\n\nprint(table(mac$cellType, mac$TYROBP_group))\n\n############################################################\n## 6. Seurat \u521d\u6b65\u53ef\u89c6\u5316\n############################################################\n\np0_1 <- Di",
  "mPlot(\n  mac,\n  reduction = \"umap\",\n  group.by = \"cellType\",\n  cols = c(\"M1 macrophages\" = \"#00A087\", \"M2 macrophages\" = \"#4DBBD5\"),\n  label = TRUE,\n  repel = TRUE\n) +\n  theme_classic(base_size = 14) +\n  ggtitle(\"Macrophage subsets\")\n\np0_2 <- FeaturePlot(\n  mac,\n  features = \"TYROBP\",\n  reduction = \"umap\"\n) +\n  theme_classic(base_size = 14) +\n  ggtitle(\"TYROBP expression\")\n\np0_3 <- DimPlot(\n  mac,\n  reduction = \"umap\",\n  group.by = \"TYROBP_group\",\n  cols = c(\"TYROBP_low\" = \"#91D1C2\", \"TYROBP_high\" = \"#DC0000\")\n) +\n  theme_classic(base_size = 14) +\n  ggtitle(\"TYROBP group\")\n\npdf(file.path(outdir, \"01_Seurat_UMAP_M1_M2.pdf\"), width = 15, height = 5)\nprint(p0_1 + p0_2 + p0_3)\ndev.off()\n\n############################################################\n## 7. \u6784\u5efa Monocle3 CellDataSet\n############################################################\n\nget_counts_safe <- function(obj, assay_use) {\n  counts_mat <- tryCatch(\n    {\n      GetAssayData(obj, assay = assay_use, layer = \"counts\")\n    },\n    error = function(e) {\n      GetAssayData(obj, assay = assay_use, slot = \"counts\")\n    }\n  )\n  return(counts_mat)\n}\n\nexpr_mat <- get_counts_safe(mac, DefaultAssay(mac))\n\ncell_metadata <- mac@meta.data\ngene_metadata <- data.frame(\n  gene_short_name = rownames(expr_mat),\n  row.names = rownames(expr_mat)\n)\n\ncds <- new_cell_data_set(\n  expression_data = expr_mat,\n  cell_metadata = cell_metadata,\n  gene_metadata = gene_metadata\n)\n\n############################################################\n## 8. Monocle3 \u8f68\u8ff9\u5206\u6790\n############################################################\n\ncds <- preprocess_cds(cds, num_d",
  "im = 30, method = \"PCA\")\n\nif (\"orig.ident\" %in% colnames(colData(cds)) &&\n    length(unique(colData(cds)$orig.ident)) > 1) {\n  cds <- align_cds(cds, alignment_group = \"orig.ident\")\n}\n\ncds <- reduce_dimension(cds, reduction_method = \"UMAP\")\ncds <- cluster_cells(cds, reduction_method = \"UMAP\")\ncds <- learn_graph(cds, use_partition = FALSE)\n\n############################################################\n## 9. \u8bbe\u5b9a root\uff1aM2 \u4e14 TYROBP \u4f4e\u8868\u8fbe\n############################################################\n\ncds$cellType <- as.character(cds$cellType)\n\nroot_cells <- colnames(cds)[\n  cds$cellType == \"M2 macrophages\" &\n    cds$TYROBP_expr <= quantile(cds$TYROBP_expr[cds$cellType == \"M2 macrophages\"], 0.5, na.rm = TRUE)\n]\n\nif (length(root_cells) < 10) {\n  root_cells <- colnames(cds)[\n    cds$cellType == \"M2 macrophages\"\n  ]\n}\n\ncat(\"root \u7ec6\u80de\u6570\uff1a\", length(root_cells), \"\\n\")\n\ncds <- order_cells(cds, root_cells = root_cells)\n\n############################################################\n## 10. \u63d0\u53d6 pseudotime \u5e76\u56de\u5199\n############################################################\n\nmac$pseudotime <- pseudotime(cds)\n\nplot_df <- mac@meta.data %>%\n  rownames_to_column(\"cell\") %>%\n  filter(!is.na(pseudotime), is.finite(pseudotime))\n\n############################################################\n## 11. \u63d0\u53d6 UMAP \u5750\u6807\n############################################################\n\numap_df <- as.data.frame(reducedDims(cds)$UMAP)\ncolnames(umap_df) <- c(\"UMAP_1\", \"UMAP_2\")\numap_df$cell <- rownames(umap_df)\n\nplot_df <- left_join(plot_df, umap_df, by = \"cell\")\n\n############################################################\n## 12. \u7f8e\u5316\u4e3b\u56fe\n",
  "############################################################\n\ncelltype_cols <- c(\n  \"M1 macrophages\" = \"#D55E00\",\n  \"M2 macrophages\" = \"#0072B2\"\n)\n\ntyrobp_cols <- c(\n  \"TYROBP_low\" = \"#91D1C2\",\n  \"TYROBP_high\" = \"#DC0000\"\n)\n\n## A. \u8f68\u8ff9\u5206\u5e03\uff08cellType\uff09\np1 <- ggplot(plot_df, aes(UMAP_1, UMAP_2)) +\n  geom_point(aes(color = cellType), size = 0.8, alpha = 0.9) +\n  scale_color_manual(values = celltype_cols) +\n  theme_classic(base_size = 14) +\n  labs(title = \"Macrophage subsets\", x = \"UMAP 1\", y = \"UMAP 2\", color = NULL) +\n  theme(\n    plot.title = element_text(face = \"bold\", hjust = 0.5),\n    legend.position = \"right\"\n  )\n\n## B. pseudotime\np2 <- ggplot(plot_df, aes(UMAP_1, UMAP_2)) +\n  geom_point(aes(color = pseudotime), size = 0.8, alpha = 0.9) +\n  scale_color_viridis_c(option = \"plasma\", name = \"Pseudotime\") +\n  theme_classic(base_size = 14) +\n  labs(title = \"Pseudotime trajectory\", x = \"UMAP 1\", y = \"UMAP 2\") +\n  theme(\n    plot.title = element_text(face = \"bold\", hjust = 0.5)\n  )\n\n## C. TYROBP along pseudotime\np3 <- ggplot(plot_df, aes(x = pseudotime, y = TYROBP_expr)) +\n  geom_point(aes(color = cellType), size = 0.8, alpha = 0.45) +\n  geom_smooth(method = \"loess\", se = TRUE, color = \"black\", linewidth = 1.1) +\n  scale_color_manual(values = celltype_cols) +\n  theme_classic(base_size = 14) +\n  labs(title = \"TYROBP along pseudotime\", x = \"Pseudotime\", y = \"TYROBP expression\", color = NULL) +\n  theme(\n    plot.title = element_text(face = \"bold\", hjust = 0.5),\n    legend.position = \"right\"\n  )\n\n## D. TYROBP-high vs low pseudotime\np4 <- ggplot(plot_df, aes(x = TYROBP_group, y = pseudoti",
  "me, fill = TYROBP_group)) +\n  geom_violin(trim = FALSE, width = 0.8, alpha = 0.85, color = \"black\") +\n  geom_boxplot(width = 0.15, outlier.shape = NA, alpha = 0.9) +\n  geom_jitter(width = 0.12, size = 0.3, alpha = 0.25) +\n  stat_compare_means(method = \"wilcox.test\", label = \"p.format\", size = 4) +\n  scale_fill_manual(values = tyrobp_cols) +\n  theme_classic(base_size = 14) +\n  labs(title = \"Pseudotime by TYROBP group\", x = NULL, y = \"Pseudotime\") +\n  theme(\n    plot.title = element_text(face = \"bold\", hjust = 0.5),\n    legend.position = \"none\"\n  )\n\n############################################################\n## 13. \u7ec4\u5408\u4e3b\u56fe\uff082 x 2\uff09\n############################################################\n\nfig_main <- (p1 + p2) / (p3 + p4) +\n  plot_annotation(tag_levels = \"A\") &\n  theme(plot.tag = element_text(size = 16, face = \"bold\"))\n\npdf(file.path(outdir, \"02_TYROBP_M1_M2_pseudotime_main_figure.pdf\"),\n    width = 12, height = 10)\nprint(fig_main)\ndev.off()\n\nggsave(\n  filename = file.path(outdir, \"02_TYROBP_M1_M2_pseudotime_main_figure.tiff\"),\n  plot = fig_main,\n  width = 12, height = 10,\n  dpi = 600,\n  compression = \"lzw\"\n)\n\n############################################################\n## 14. TYROBP \u5728\u8f68\u8ff9\u76f8\u5173\u57fa\u56e0\u4e2d\u7684\u53d8\u5316\n############################################################\n\ngenes_use <- c(\n  \"TYROBP\", \"FCER1G\", \"LILRB3\", \"GNA15\",\n  \"IL1B\", \"TNF\", \"CCL3\", \"CCL4\",\n  \"SPP1\", \"CD68\", \"LYZ\", \"MRC1\",\n  \"CD163\", \"C1QA\", \"C1QB\", \"TREM2\"\n)\n\ngenes_use <- intersect(genes_use, rownames(cds))\n\nif (length(genes_use) > 0) {\n  pdf(file.path(outdir, \"03_marker_genes_in_pseudotime.pdf\"), width = 12, height = 8)",
  "\n  print(\n    plot_genes_in_pseudotime(\n      cds[genes_use, ],\n      color_cells_by = \"cellType\",\n      min_expr = 0.1,\n      ncol = 3\n    )\n  )\n  dev.off()\n}\n\n############################################################\n## 15. \u62df\u65f6\u5e8f\u76f8\u5173\u57fa\u56e0\u7b5b\u9009\n############################################################\n\ndeg_pseudo <- graph_test(\n  cds,\n  neighbor_graph = \"principal_graph\",\n  cores = 4\n)\n\ndeg_pseudo <- deg_pseudo %>%\n  arrange(q_value)\n\nwrite.csv(\n  deg_pseudo,\n  file.path(outdir, \"04_genes_associated_with_pseudotime.csv\"),\n  row.names = TRUE\n)\n\n############################################################\n## 16. \u4fdd\u5b58\u5bf9\u8c61\n############################################################\n\nsaveRDS(mac, file.path(outdir, \"mac_M1_M2_with_pseudotime.rds\"))\nsaveRDS(cds, file.path(outdir, \"cds_M1_M2_with_pseudotime.rds\"))\n\ncat(\"\u5206\u6790\u5b8c\u6210\uff0c\u7ed3\u679c\u8f93\u51fa\u5230\uff1a\", outdir, \"\\n\")\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
)

.CELLCHAT_ORIGINAL_LAYOUT <- paste0(
  "# Dimensions are in inches for the page and in centimetres for the two\n# CellChat heatmap bodies. They determine layout only.\npage_width <- 14.35\npage_height <- 15.15\nheat_pair_width <- 7.95\nheat_pair_height <- 6.55\nheat_body_width <- 5.75\nheat_body_height <- 11.40\ntrajectory_row_width <- page_width\ntrajectory_row_height <- 4.55\n\nplot_box <- function(x, top, width, height, parent_height) {\n  grid::viewport(x=grid::unit(x, \"inches\"), y=grid::unit(parent_height-top, \"inches\"),\n                 width=grid::unit(width, \"inches\"), height=grid::unit(height, \"inches\"),\n                 just=c(\"left\",\"top\"))\n}\n\npanel_letter <- function(letter, x, top, parent_height) {\n  grid::grid.text(letter, x=grid::unit(x,\"inches\"), y=grid::unit(parent_height-top,\"inches\"),\n                  just=c(\"left\",\"top\"), gp=grid::gpar(fontsize=18.5,fontface=\"plain\"))\n}\n\ndraw_grob_at <- function(grob, x, top, width, height, parent_height) {\n  grid::pushViewport(plot_box(x,top,width,height,parent_height))\n  grid::grid.draw(grob)\n  grid::upViewport()\n}\n\ndraw_heat_pair <- function() {\n  half <- heat_pair_width/2\n  for (index in 1:2) {\n    pattern <- c(\"outgoing\",\"incoming\")[index]\n    x <- (index-1)*half\n    h <- role_heatmap(pattern, width_cm=heat_body_width, height_cm=heat_body_height,\n                      font_size=9.0)\n    grid::pushViewport(plot_box(x+.04,.30,half-.08,heat_pair_height-.36,heat_pair_height))\n    ComplexHeatmap::draw(h,newpage=FALSE)\n    grid::upViewport()\n    panel_letter(c(\"D\",\"E\")[index], x+.08, .075, heat_pair_height)\n  }\n}\n\n# Move the original legends below the axes to free horizon",
  "tal plotting space.\n# F and G have the same coordinates, limits and physical units on both axes.\ndisplay_theme <- ggplot2::theme(\n  plot.title=ggplot2::element_text(size=14),\n  axis.title=ggplot2::element_text(size=12),\n  axis.text=ggplot2::element_text(size=10),\n  legend.title=ggplot2::element_text(size=10.5),\n  legend.text=ggplot2::element_text(size=10.5),\n  legend.position=\"bottom\",\n  legend.box.margin=ggplot2::margin(0,0,0,0),\n  legend.margin=ggplot2::margin(0,0,0,0),\n  plot.margin=ggplot2::margin(5,6,3,5)\n)\numap_x <- range(plot_df$UMAP_1)\numap_y <- range(plot_df$UMAP_2)\numap_aspect <- diff(umap_y)/diff(umap_x)\ncommon_umap <- ggplot2::coord_fixed(ratio=1,xlim=umap_x,ylim=umap_y)\ntrajectory_plots <- list(\n  p_states + display_theme + common_umap + ggplot2::guides(colour=ggplot2::guide_legend(nrow=1)),\n  p_time + display_theme + common_umap +\n    ggplot2::guides(colour=ggplot2::guide_colourbar(\n      display=\"gradient\",direction=\"horizontal\",\n      theme=ggplot2::theme(legend.title.position=\"top\",\n                          legend.title=ggplot2::element_text(hjust=.5,margin=ggplot2::margin(b=2)),\n                          legend.key.width=grid::unit(1.55,\"inches\"),\n                          legend.key.height=grid::unit(.13,\"inches\")))),\n  p_expression + display_theme + ggplot2::theme(aspect.ratio=umap_aspect) +\n    ggplot2::guides(colour=ggplot2::guide_legend(nrow=1))\n)\n\ndraw_trajectory_row <- function() {\n  # Draw the original guides independently from their axes so the horizontal\n  # colour bar and the two categorical legends do not change panel height.\n  legends <- lapp",
  "ly(trajectory_plots, function(p) cowplot::get_legend(p,legend=\"bottom\"))\n  panels <- lapply(trajectory_plots, function(p) ggplot2::ggplotGrob(p+ggplot2::theme(legend.position=\"none\")))\n  col_widths <- do.call(grid::unit.pmax, lapply(panels, function(g) g$widths))\n  row_heights <- do.call(grid::unit.pmax, lapply(panels, function(g) g$heights))\n  column <- trajectory_row_width/3\n  for (index in seq_along(panels)) {\n    g <- panels[[index]]\n    g$widths <- col_widths\n    g$heights <- row_heights\n    x <- (index-1)*column\n    draw_grob_at(g, x+.07,.28,column-.14,3.69,trajectory_row_height)\n    draw_grob_at(legends[[index]], x+.05,3.99,column-.10,.50,trajectory_row_height)\n    panel_letter(c(\"F\",\"G\",\"H\")[index], x+.12,.055,trajectory_row_height)\n  }\n}\n\nsave_vector_pdf <- function(filename,width,height,draw) {\n  grDevices::cairo_pdf(file.path(project_dir,filename),width=width,height=height,family=\"sans\")\n  on.exit(grDevices::dev.off(),add=TRUE)\n  force(draw)\n  invisible(NULL)\n}\n\ndraw_network_row <- function(metrics,width,height) {\n  directions <- c(\"out\",\"in\")\n  circles <- expand.grid(direction=directions,metric=metrics,stringsAsFactors=FALSE)\n  column <- width/nrow(circles)\n  for (index in seq_len(nrow(circles))) {\n    x <- (index-1)*column+.06\n    par(fig=c(x/width,(x+column-.12)/width,1-3.04/height,1-.25/height),\n        mar=c(.05,.1,.05,.1), new=index>1,xpd=NA)\n    network(circles$metric[index],circles$direction[index])\n  }\n  letters <- c(count=\"A\",strength=\"B\")\n  for (index in seq_along(metrics)) {\n    panel_letter(letters[[metrics[index]]],(index-1)*2*column+.12,.095,height",
  ")\n  }\n}\n\ndraw_full_figure <- function() {\n  draw_network_row(c(\"count\",\"strength\"),page_width,page_height)\n  heat_top <- 3.27\n  print(p_bubble,vp=plot_box(.11,3.48,6.20,7.04,page_height))\n  panel_letter(\"C\",.12,heat_top+.075,page_height)\n\n  grid::pushViewport(plot_box(6.36,heat_top,heat_pair_width,heat_pair_height,page_height))\n  draw_heat_pair()\n  grid::upViewport()\n  grid::pushViewport(plot_box(0,10.55,trajectory_row_width,trajectory_row_height,page_height))\n  draw_trajectory_row()\n  grid::upViewport()\n}\n\nfull_pdf <- \"Figure7_original_strength.pdf\"\npanel_b_pdf <- \"Figure7B_original_strength.pdf\"\nsave_vector_pdf(full_pdf,page_width,page_height,draw_full_figure())\nsave_vector_pdf(panel_b_pdf,page_width/2,3.04,draw_network_row(\"strength\",page_width/2,3.04))\n\ninvisible(pdftools::pdf_convert(file.path(project_dir,full_pdf),format=\"png\",pages=1,dpi=600,\n  filenames=file.path(project_dir,\"Figure7_original_strength_600dpi.png\"),verbose=FALSE))\ninvisible(pdftools::pdf_convert(file.path(project_dir,panel_b_pdf),format=\"png\",pages=1,dpi=150,\n  filenames=file.path(project_dir,\"Figure7B_preview.png\"),verbose=FALSE))\ncapture.output(sessionInfo(),file=file.path(project_dir,\"R_session.txt\"))\nmessage(\"Figure 7 and the individual panel B are in \",project_dir)\n"
)


# Donor pseudobulk and bulk-adjusted expression: native original R scripts.
.ROOT_R_SOURCES <- list(
bulk_adjusted=paste0(
"# Style adapted from TYROBP_publication_violins_20261112.R.\n# Data/model from the original bulk macrophage ssGSEA adjustment, not scRNA-seq.\n# A real adjusted value is derived for every sample: original OLS residual plus\n# the fitted mean at that sample's group and the pooled common covariate profile.\nsuppressPackageStartupMessages({library(ggplot2);library(jsonlite)})\n# Run: Rscript code/Redraw_bulk_adjusted_TYROBP_with_title.R [package_directory]\nargs <- commandArgs(TRUE)\nscript_arg <- grep('^--file=',commandArgs(FALSE),value=TRUE)\ndefault_root <- dirname(dirname(normalizePath(sub('^--file=','',script_arg[1]),mustWork=TRUE)))\nout <- if(length(args)) normalizePath(args[1],mustWork=TRUE) else default_root\nsrc <- file.path(out,'source_data')\nfor(d in c('figures','previews','source_data','tables','code','documentation'))dir.create(file.path(out,d),recursive=TRUE,showWarnings=FALSE)\ndata_path<-file.path(src,'sample_scores_and_expression.csv')\neffect_path<-file.path(src,'ssGSEA_adjusted_effects.csv')\nmeans_path<-file.path(src,'model_adjusted_expression_means.csv')\nd<-read.csv(data_path,check.names=FALSE,fileEncoding='UTF-8-BOM')\nd<-d[d$dataset=='Original_ComBat_pooled',,drop=FALSE]\nrownames(d)<-NULL\nd$group<-factor(d$group,levels=c('Control','UC'))\nd$cohort<-factor(d$cohort,levels=c('GSE206285','GSE87466'))\nstopifnot(nrow(d)==312,!anyDuplicated(d$sample),sum(d$group=='Control')==39,\n sum(d$group=='UC')==273,all(d$uc==as.integer(d$group=='UC')),\n all(is.finite(d$TYROBP)),all(is.finite(d$ssgsea_z)))\nold_effect<-read.csv(effect_path,fileEncoding='UTF-8-BOM')\nold_effect<-old_effect[old_effect$dataset=='Original_ComBat_pooled' & old_effect$adjustment=='Macrophage_ssGSEA',,drop=FAL",
"SE]\nold_means<-read.csv(means_path,fileEncoding='UTF-8-BOM')\n\n# Reproduce existing additive OLS and HC3 inference without changing covariates.\nfit<-lm(TYROBP~uc+cohort+ssgsea_z,data=d)\nX<-model.matrix(fit);b<-coef(fit);res<-residuals(fit)\nV0<-solve(crossprod(X));h<-hatvalues(fit)\nXw<-sweep(X,1,res/(1-h),'*')\nVhc3<-V0%*%crossprod(Xw)%*%V0\nuc_index<-match('uc',colnames(X))\nse<-sqrt(Vhc3[uc_index,uc_index]);p<-2*pt(-abs(b['uc']/se),df.residual(fit))\nci<-b['uc']+c(-1,1)*qt(.975,df.residual(fit))*se\nstopifnot(abs(b['uc']-old_effect$beta_UC)<1e-10,\n abs(p-old_effect$p_UC)<1e-10,abs(se-old_effect$se_HC3)<1e-10)\nxbar<-colMeans(X)\nxref<-matrix(rep(xbar,each=nrow(d)),nrow=nrow(d))\ncolnames(xref)<-colnames(X);xref[,'uc']<-d$uc\nd$model_residual<-as.numeric(res)\nd$standardized_fitted_group_mean<-as.numeric(xref%*%b)\nd$adjusted_TYROBP<-d$model_residual+d$standardized_fitted_group_mean\n# Equivalent removal of sample-specific nuisance contributions, without removing uc.\nnuisance<-setdiff(seq_len(ncol(X)),c(match('(Intercept)',colnames(X)),uc_index))\nby_nuisance<-d$TYROBP-as.numeric(X[,nuisance,drop=FALSE]%*%b[nuisance])+sum(xbar[nuisance]*b[nuisance])\nstopifnot(max(abs(by_nuisance-d$adjusted_TYROBP))<1e-10)\ngm<-aggregate(adjusted_TYROBP~group,d,mean)\ntarget<-old_means$adjusted_mean[match(gm$group,old_means$group)]\nstopifnot(max(abs(gm$adjusted_TYROBP-target))<1e-10)\nwrite.csv(d,file.path(out,'tables/Adjusted_sample_values.csv'),row.names=FALSE)\nwrite.csv(data.frame(group=gm$group,adjusted_mean=gm$adjusted_TYROBP,original_adjusted_mean=target),\n file.path(out,'tables/Adjusted_mean_reproduction.csv'),row.names=FALSE)\nwrite.csv(data.frame(beta_UC=b['uc'],se_HC3=se,df=df.residual(fit),P=p,ci",
"95_lower=ci[1],ci95_upper=ci[2]),\n file.path(out,'tables/Original_model_reproduction.csv'),row.names=FALSE)\nsaveRDS(list(fit=fit,HC3_covariance=Vhc3,common_profile=xbar,plotted_data=d),file.path(out,'documentation/Adjusted_violin_model.rds'))\n\npalette<-c(Control='#173F68',UC='#A51C39')\nspan<-diff(range(d$adjusted_TYROBP));bracket<-max(d$adjusted_TYROBP)+.08*span\nlimits<-c(min(d$adjusted_TYROBP)-.07*span,bracket+.14*span)\nset.seed(20261114)\nd$x_dot<-as.numeric(d$group)+runif(nrow(d),-.18,.18)\npplot<-ggplot(d,aes(x=as.numeric(group),y=adjusted_TYROBP,group=group))+\n geom_violin(aes(fill=group,colour=group),width=.74,trim=TRUE,scale='width',adjust=.95,linewidth=.55,show.legend=FALSE)+\n geom_point(aes(x=x_dot,fill=group),shape=21,colour='white',size=1.05,stroke=.22,alpha=.78,show.legend=FALSE)+\n geom_boxplot(width=.105,outlier.shape=NA,fill='white',colour='#18232D',linewidth=.48,show.legend=FALSE)+\n annotate('segment',x=1,xend=2,y=bracket,yend=bracket,linewidth=.5,colour='#252D34')+\n annotate('segment',x=c(1,2),xend=c(1,2),y=bracket,yend=bracket-.026*diff(limits),linewidth=.5,colour='#252D34')+\n annotate('text',x=1.5,y=bracket+.048*diff(limits),label=sprintf('italic(P) == %.2f',p),parse=TRUE,size=4.1,family='Arial',colour='#17212B')+\n scale_fill_manual(values=palette)+scale_colour_manual(values=palette)+\n scale_x_continuous(breaks=c(1,2),labels=c('Control\\n(n = 39)','UC\\n(n = 273)'),limits=c(.45,2.55),expand=c(0,0))+\n scale_y_continuous(breaks=pretty(limits,n=5),labels=function(x)formatC(x,format='f',digits=1),expand=c(0,0))+\n coord_cartesian(ylim=limits,clip='on')+\n labs(x=NULL,y=expression('Adjusted TYROBP expression (log'[2]*')'),\n      title='Macrophage score-adjusted TYR",
"OBP')+\n theme_classic(base_family='Arial',base_size=11)+\n theme(plot.title=element_text(size=11.5,face='plain',hjust=.5,colour='#17212B',margin=margin(b=12)),\n  axis.title.y=element_text(size=11.5,margin=margin(r=8)),\n  axis.text=element_text(size=11,colour='#17212B'),\n  axis.text.x=element_text(lineheight=.95,margin=margin(t=6)),\n  axis.line=element_line(colour='#252D34',linewidth=.55),\n  axis.ticks=element_line(colour='#252D34',linewidth=.5),\n  axis.ticks.length=grid::unit(3,'pt'),legend.position='none',\n  plot.margin=margin(11,14,10,10))\nplots<-list(TYROBP_Adjusted_Violin_with_title=pplot)\n# Verify all original adjusted values and horizontal point positions.\nprevious <- read.csv(file.path(src,'previous_Adjusted_sample_values_with_jitter.csv'))\nii <- match(d$sample,previous$sample)\nstopifnot(!anyNA(ii),max(abs(d$adjusted_TYROBP-previous$adjusted_TYROBP[ii]))<1e-10,\n          max(abs(d$x_dot-previous$x_dot[ii]))<1e-10)\nfor(stem in names(plots)){\n q<-plots[[stem]];prefix<-file.path(out,'figures',stem)\n ggsave(paste0(prefix,'.pdf'),q,width=3.95,height=4.05,device=cairo_pdf,family='Arial',bg='white')\n ggsave(paste0(prefix,'.png'),q,width=3.95,height=4.05,dpi=600,device=ragg::agg_png,bg='white')\n ggsave(paste0(prefix,'.svg'),q,width=3.95,height=4.05,device=svglite::svglite,bg='white')\n ggsave(paste0(prefix,'.tiff'),q,width=3.95,height=4.05,dpi=600,device=ragg::agg_tiff,compression='lzw',bg='white')\n ggsave(file.path(out,'previews',paste0(stem,'.png')),q,width=3.95,height=4.05,dpi=160,device=ragg::agg_png,bg='white')\n}\nsaveRDS(plots,file.path(out,'documentation/violin_plots.rds'))\nwrite.csv(d,file.path(out,'tables/Adjusted_sample_values_with_jitter.csv'),row.names=FALSE)\n# Pa",
"ckaged input retained in place.\n# Packaged input retained in place.\n# Packaged input retained in place.\njsonlite::write_json(list(n=312,n_Control=39,n_UC=273,p_HC3=p,beta_UC=unname(b['uc']),\n adjusted_mean_errors=unname(gm$adjusted_TYROBP-target),formula_equivalence_max_error=max(abs(by_nuisance-d$adjusted_TYROBP)),\n raw_expression_unchanged=TRUE,no_simulated_data=TRUE),file.path(out,'documentation/validation.json'),pretty=TRUE,auto_unbox=TRUE,digits=15)\ncapture.output(sessionInfo(),file=file.path(out,'code/sessionInfo.txt'))\ncat('Output:',out,'\\nP=',p,'\\nbeta=',b['uc'],'\\n');print(gm);print(aggregate(adjusted_TYROBP~group,d,range))\n"
),
macrophage_pseudobulk=paste0(
"# Refit from packaged raw donor-by-population UMI sums.\n# Run from the package folder: Rscript code/01_refit_pseudobulk.R .\n# The supplied 33538-gene matrices were reconstructed from downloaded 10X files\n# on 2026-11-16; existing QC barcodes and current annotations were retained.\nsuppressPackageStartupMessages(library(edgeR))\nargs <- commandArgs(TRUE)\nout <- if (length(args)) normalizePath(args[1], mustWork=TRUE) else getwd()\nset.seed(20261116)\nfor (d in c('tables','models')) dir.create(file.path(out,d),showWarnings=FALSE)\nscopes <- c('C1Q_mature','ACOD1_inflammatory_like','NRG1_activated_like',\n            'Combined_macrophage_like')\ninferential_scopes <- c('C1Q_mature','Combined_macrophage_like')\nmap <- read.csv(file.path(out,'inputs/gene_id_mapping.csv'))\nresults <- list(); expressions <- list(); coverage <- list()\n\nfor (scope in scopes) {\n  pb <- as.matrix(read.csv(file.path(out,'inputs',paste0(scope,'_raw_counts.csv')),\n                          row.names=1,check.names=FALSE))\n  sm0 <- read.csv(file.path(out,'inputs',paste0(scope,'_samples.csv')))\n  stopifnot(nrow(pb)==33538L, ncol(pb)==12L,\n            identical(colnames(pb),sm0$donor),\n            all(pb>=0), all(pb==round(pb)),\n            all(colSums(pb)==sm0$all_gene_UMI),\n            all(pb['TYROBP',]==sm0$TYROBP_UMI))\n  cv <- sm0\n  cv$coverage_status <- ifelse(cv$n_cells==0,'no_cells_expression_missing','observed')\n  cv$expression_eligible_all_observed <- cv$n_cells>=1\n  cv$expression_eligible_min10_cells <- cv$n_cells>=10\n  coverage[[scope]] <- cv\n\n  for (mode in c('all_observed','min10_cells')) {\n    threshold <- if (mode=='all_observed') 1 else 10\n    sm <- sm0[sm0$n_cells>=threshold,,drop=FALSE]\n    sm$gro",
"up <- factor(sm$group, levels=c('Control','UC'))\n    rownames(sm) <- sm$donor\n    ns <- table(sm$group)\n    # Same inferential eligibility as the previous analysis for these scopes.\n    # Sparse scopes are not promoted to inferential results by plotting them.\n    allowed <- scope %in% inferential_scopes && all(ns>=3)\n    reason <- if (allowed) 'tested' else if (ns['Control']==0)\n      'not_tested_no_control_donors_with_cells' else\n      'not_tested_insufficient_donor_coverage'\n    note <- if (allowed)\n      'TMM after filterByExpr; retained-gene library sizes; prior.count=2' else\n      'Descriptive TMM across all 33538 measured genes; no expression filter; prior.count=2; no group test'\n    r <- data.frame(scope=scope, analysis=mode,min_cells=threshold,\n                    Control_donors=unname(ns['Control']),UC_donors=unname(ns['UC']),\n                    Control_cells=sum(sm$n_cells[sm$group=='Control']),\n                    UC_cells=sum(sm$n_cells[sm$group=='UC']),\n                    status=reason,testable=allowed,\n                    logFC=NA_real_,logCPM=NA_real_,F=NA_real_,\n                    PValue=NA_real_,FDR=NA_real_,genes_tested=NA_integer_,\n                    normalization_note=note)\n    if (nrow(sm)==0) {\n      results[[paste(scope,mode)]] <- r\n      next\n    }\n    y <- DGEList(counts=pb[,sm$donor,drop=FALSE],samples=sm)\n    if (allowed) {\n      design <- model.matrix(~group,sm)\n      keep <- filterByExpr(y,design=design)\n      y <- y[keep,,keep.lib.sizes=FALSE]\n      y <- calcNormFactors(y,method='TMM')\n      y <- estimateDisp(y,design,robust=TRUE)\n      fit <- glmQLFit(y,design,robust=TRUE)\n      test <- glmQLFTest(fit,coef='groupUC')\n      rr <- topTags(",
"test,n=Inf,sort.by='none')$table\n      rr$gene <- rownames(rr);rr$scope <- scope;rr$analysis <- mode\n      rr$ensembl_id <- map$ensembl_id[match(rr$gene,map$gene)]\n      stopifnot(isTRUE(all.equal(rr$FDR,p.adjust(rr$PValue,method='BH'))))\n      write.csv(rr,file.path(out,'tables',paste0(scope,'_',mode,'_all_genes.csv')),\n                row.names=FALSE)\n      r$genes_tested <- nrow(y)\n      if ('TYROBP' %in% rownames(rr)) {\n        for (z in c('logFC','logCPM','F','PValue','FDR')) r[[z]] <- rr['TYROBP',z]\n      } else {\n        r$testable <- FALSE;r$status <- 'not_tested_filtered_low_expression'\n      }\n      saveRDS(list(raw_counts=pb[,sm$donor,drop=FALSE],samples=sm,\n                   design=design,DGE=y,fit=fit,test=test),\n              file.path(out,'models',paste0(scope,'_',mode,'_edgeR.rds')))\n    } else {\n      # No dispersion/model fitting in sparse or absent-control populations.\n      y <- calcNormFactors(y,method='TMM')\n      saveRDS(list(raw_counts=pb[,sm$donor,drop=FALSE],samples=sm,DGE=y,\n                   note=note),\n              file.path(out,'models',paste0(scope,'_',mode,'_descriptive_TMM.rds')))\n    }\n    lcpm <- cpm(pb[,sm$donor,drop=FALSE],\n                lib.size=y$samples$lib.size*y$samples$norm.factors,\n                log=TRUE,prior.count=2)\n    ex <- sm\n    ex$gene <- 'TYROBP'\n    ex$TMM_log2CPM <- as.numeric(lcpm['TYROBP',])\n    ex$normalization_factor <- y$samples$norm.factors\n    ex$library_size_for_normalization <- y$samples$lib.size\n    ex$effective_library_size <- y$samples$lib.size*y$samples$norm.factors\n    ex$analysis <- mode\n    ex$testable <- r$testable\n    ex$normalization_note <- note\n    expressions[[paste(scope,mode)]] <- ex\n   ",
" results[[paste(scope,mode)]] <- r\n    cat(scope,mode,'n=',paste(as.integer(ns),collapse='/'),\n        'P=',r$PValue,'FDR=',r$FDR,'\\n')\n  }\n}\n\nrs <- do.call(rbind,results)\nex <- do.call(rbind,expressions)\ncv <- do.call(rbind,coverage)\nrownames(rs)<-NULL;rownames(ex)<-NULL;rownames(cv)<-NULL\nwrite.csv(rs[rs$analysis=='all_observed',],\n          file.path(out,'tables/TYROBP_scope_results.csv'),row.names=FALSE,na='NA')\nwrite.csv(rs[rs$analysis=='min10_cells',],\n          file.path(out,'tables/TYROBP_scope_results_min10_cells.csv'),row.names=FALSE,na='NA')\nwrite.csv(ex[ex$analysis=='all_observed',],\n          file.path(out,'tables/TYROBP_donor_expression.csv'),row.names=FALSE,na='NA')\nwrite.csv(ex[ex$analysis=='min10_cells',],\n          file.path(out,'tables/TYROBP_donor_expression_min10_cells.csv'),row.names=FALSE,na='NA')\nwrite.csv(cv,file.path(out,'tables/TYROBP_donor_coverage_all12.csv'),row.names=FALSE,na='NA')\n\n# Reproducibility check against the 2026-11-16 patient-level analysis.\nold <- read.csv(file.path(out,'inputs/previous_TYROBP_results_20261116.csv'))\nchecked <- merge(rs[rs$testable,],old,by=c('scope','analysis'),suffixes=c('_new','_previous'))\nchecked$abs_P_difference <- abs(checked$PValue_new-checked$PValue_previous)\nchecked$abs_FDR_difference <- abs(checked$FDR_new-checked$FDR_previous)\nchecked$abs_logFC_difference <- abs(checked$logFC_new-checked$logFC_previous)\nstopifnot(nrow(checked)==sum(rs$testable), all(checked$abs_P_difference<1e-10),\n          all(checked$abs_FDR_difference<1e-10),all(checked$abs_logFC_difference<1e-10))\nwrite.csv(checked[,c('scope','analysis','PValue_new','PValue_previous',\n                    'FDR_new','FDR_previous','logFC_new','logF",
"C_previous',\n                    'abs_P_difference','abs_FDR_difference','abs_logFC_difference')],\n          file.path(out,'tables/refit_verification.csv'),row.names=FALSE)\noldex <- read.csv(file.path(out,'inputs/previous_TYROBP_donor_expression_20261116.csv'))\nce <- merge(ex[ex$testable,],oldex,by=c('scope','analysis','donor'),suffixes=c('_new','_previous'))\nstopifnot(nrow(ce)==sum(ex$testable),all(abs(ce$TMM_log2CPM_new-ce$TMM_log2CPM_previous)<1e-10))\ncapture.output(sessionInfo(),file=file.path(out,'tables/pseudobulk_sessionInfo.txt'))\ncat('Validated: 4 model results and 46 modeled donor-expression values reproduce the prior analysis.\\n')\n"
),
donor_pseudobulk=paste0(
"# Run from the analysis directory: Rscript --vanilla pseudobulk.R\n# A sample column is one independent donor; UC minus healthy control.\nsuppressPackageStartupMessages(library(edgeR))\ndir.create(\"results\",showWarnings=FALSE)\nset.seed(20261116)\nmajor <- c(\"B_Plasma\",\"T_cells\",\"Epithelial\",\"Stromal\",\"Myeloid\")\nsub <- c(\"C1Q_mature\",\"ACOD1_inflammatory_like\",\"NRG1_activated_like\")\npool <- \"Combined_macrophage_like\"\nscopes <- c(pool,sub,\"Myeloid\",\"All_captured\")\nmat <- list(); sm <- list()\nread_inputs <- function(scope){\n  src <- file.path(\"inputs\",\"counts\",scope)\n  x <- as.matrix(read.csv(paste0(src,\"_raw_counts.csv\"),row.names=1,check.names=FALSE))\n  meta <- read.csv(paste0(src,\"_samples.csv\"),stringsAsFactors=FALSE)\n  stopifnot(nrow(x)==33538,ncol(x)==12,!anyDuplicated(rownames(x)),\n            identical(colnames(x),meta$donor),min(x)>=0,\n            all(x==round(x)),all(colSums(x)==meta$all_gene_UMI),\n            all(x[\"TYROBP\",]==meta$TYROBP_UMI))\n  list(x=x,meta=meta)\n}\nfor (k in unique(c(major,pool,sub))){\n  tmp <- read_inputs(k)\n  mat[[k]] <- tmp$x; sm[[k]] <- tmp$meta\n  stopifnot(identical(dimnames(mat[[k]]),dimnames(mat[[major[1]]])))\n}\nstopifnot(all(mat[[pool]]==Reduce(\"+\",mat[sub])))\nmat[[\"All_captured\"]] <- Reduce(\"+\", mat[major])\ng <- sm[[major[1]]]\nfor (col in c(\"n_cells\",\"all_gene_UMI\",\"TYROBP_UMI\")) {\n  g[[col]] <- Reduce(\"+\",lapply(sm[major], \"[[\", col))\n}\ng$scope <- \"All_captured\"; g$percent_TYROBP_positive <- NA_real_\nsm[[\"All_captured\"]] <- g\nstopifnot(sum(g$n_cells)==21965,all(colSums(mat[[\"All_captured\"]])==g$all_gene_UMI))\nstats <- list(); donors <- list()\nfor (k in scopes) for (cut in c(1,10)){\n  if(k==\"All_captured\" && cut!=1) next\n  a <- sm[[k]]\n  a ",
"<- a[a$n_cells>=cut,,drop=FALSE]\n  a$group <- factor(a$group,levels=c(\"Control\",\"UC\"));row.names(a)<-a$donor\n  ns <- table(a$group)\n  testable <- !k %in% c(\"ACOD1_inflammatory_like\",\"NRG1_activated_like\") && all(ns>=3)\n  x <- mat[[k]][,a$donor,drop=FALSE]; y <- DGEList(counts=x,samples=a)\n  stat <- data.frame(scope=k,min_cells=cut,n_HC=unname(ns[1]),n_UC=unname(ns[2]),\n       cells_HC=sum(a$n_cells[a$group==\"Control\"]),cells_UC=sum(a$n_cells[a$group==\"UC\"]),\n       testable=testable,logFC=NA_real_,PValue=NA_real_,FDR=NA_real_,genes_tested=NA_integer_)\n  if(testable){\n    design <- model.matrix(~group,a)\n    keep <- filterByExpr(y,design=design)\n    y <- y[keep,,keep.lib.sizes=FALSE]\n    y <- calcNormFactors(y,method=\"TMM\")\n    y <- estimateDisp(y,design,robust=TRUE)\n    fit <- glmQLFit(y,design,robust=TRUE)\n    tt <- glmQLFTest(fit,coef=\"groupUC\")\n    tab <- topTags(tt,n=Inf,sort.by=\"none\")$table\n    stopifnot(\"TYROBP\" %in% rownames(tab))\n    for (n in c(\"logFC\",\"PValue\",\"FDR\")) stat[[n]] <- tab[\"TYROBP\",n]\n    stat$genes_tested <- nrow(tab)\n  } else {\n    y <- calcNormFactors(y,method=\"TMM\")\n  }\n  eff <- y$samples$lib.size*y$samples$norm.factors\n  norm <- cpm(x,lib.size=eff,log=TRUE,prior.count=2)[\"TYROBP\",]\n  donors[[length(donors)+1L]] <- data.frame(scope=k,min_cells=cut,donor=a$donor,\n       group=as.character(a$group),n_cells=a$n_cells,raw_TYROBP=a$TYROBP_UMI,\n       raw_library=a$all_gene_UMI,effective_library=eff,log2CPM=unname(norm))\n  stats[[length(stats)+1L]] <- stat\n  cat(k,cut,\"; donors\",as.integer(ns),\"; QL P\",signif(stat$PValue,6),\"\\n\")\n}\nstats <- do.call(rbind,stats); donors <- do.call(rbind,donors)\nfor(cut in c(1,10)){\n suffix <- if(cut==1) \"\" else \"_min10",
"_cells\"\n prev <- read.csv(file.path(\"inputs\",\"archived\",paste0(\"TYROBP_scope_results\",suffix,\".csv\")))\n curr <- stats[stats$scope!=\"All_captured\" & stats$min_cells==cut,]\n m <- merge(curr,prev,by=\"scope\",suffixes=c(\".new\",\".old\"))\n stopifnot(nrow(m)==5, all(m$n_HC==m$Control_donors),all(m$n_UC==m$UC_donors))\n for(z in c(\"logFC\",\"PValue\",\"FDR\")){\n  a <- m[[paste0(z,\".new\")]]; b <- m[[paste0(z,\".old\")]]\n  stopifnot(identical(is.na(a),is.na(b)), max(abs(a-b),na.rm=TRUE)<1e-8)\n }\n oldex <- read.csv(file.path(\"inputs\",\"archived\",paste0(\"TYROBP_donor_expression\",suffix,\".csv\")))\n d <- donors[donors$scope!=\"All_captured\" & donors$min_cells==cut,]\n joint <- merge(d,oldex,by=c(\"scope\",\"donor\"))\n stopifnot(nrow(joint)==nrow(d),nrow(d)==nrow(oldex),max(abs(joint$log2CPM-joint$TMM_log2CPM))<1e-8)\n}\nwrite.csv(stats,\"results/expression_tests.csv\",row.names=FALSE,na=\"\")\nwrite.csv(donors,\"results/expression_observed.csv\",row.names=FALSE,na=\"\")\ncapture.output(sessionInfo(),file=\"results/R_session.txt\")\ncat(\"Archived results verified for both cell-count thresholds.\\n\")\n"
))

.root_copy_tree <- function(from,to){
  dir.create(to,recursive=TRUE,showWarnings=FALSE)
  f <- list.files(from,all.files=TRUE,no..=TRUE,full.names=TRUE)
  if(length(f)) stopifnot(all(file.copy(f,to,recursive=TRUE,overwrite=TRUE)))
}
.root_eval <- function(key,out){
  dir.create(out,recursive=TRUE,showWarnings=FALSE)
  out <- normalizePath(out,winslash='/',mustWork=TRUE)
  e <- new.env(parent=globalenv())
  e$commandArgs <- function(trailingOnly=FALSE) if(trailingOnly) out else paste0('--file=',file.path(out,'code','master.R'))
  dir.create(file.path(out,'code'),recursive=TRUE,showWarnings=FALSE)
  file.create(file.path(out,'code','master.R'))
  oldwd <- getwd();on.exit(setwd(oldwd),add=TRUE);setwd(out)
  eval(parse(text=.ROOT_R_SOURCES[[key]],keep.source=TRUE),envir=e)
  invisible(out)
}
run_bulk_adjusted <- function(data_root,output_root,...){
  out <- file.path(output_root,'fig01_L_bulk_adjusted')
  .root_copy_tree(file.path(data_root,'bulk_adjusted','source_data'),file.path(out,'source_data'))
  .root_eval('bulk_adjusted',out)
}
run_macrophage_pseudobulk <- function(data_root,output_root,...){
  out <- file.path(output_root,'analysis_macrophage_pseudobulk')
  .root_copy_tree(file.path(data_root,'macrophages','inputs'),file.path(out,'inputs'))
  .root_eval('macrophage_pseudobulk',out)
}
run_donor_pseudobulk <- function(data_root,output_root,...){
  out <- file.path(output_root,'analysis_donor_pseudobulk')
  .root_copy_tree(file.path(data_root,'donors','inputs'),file.path(out,'inputs'))
  .root_eval('donor_pseudobulk',out)
}
ROOT_R_STAGES <- list(fig01_L_bulk_adjusted=run_bulk_adjusted,
                     analysis_macrophage_pseudobulk=run_macrophage_pseudobulk,
                     analysis_donor_pseudobulk=run_donor_pseudobulk)


R_STAGES <- c(if(exists('ANNO_STAGES')) lapply(ANNO_STAGES,function(x)x$fun) else list(),CELLCHAT_STAGES,ROOT_R_STAGES)
R_DEFAULT_STAGES <- c(if(exists('ANNO_STAGES')) names(ANNO_STAGES)[vapply(ANNO_STAGES,function(x)x$default,logical(1))] else character(),names(CELLCHAT_STAGES),names(ROOT_R_STAGES))
.master_base <- function(){
  f<-grep('^--file=',commandArgs(FALSE),value=TRUE)
  if(length(f)) dirname(normalizePath(sub('^--file=','',f[1]),winslash='/',mustWork=FALSE)) else getwd()
}
run_r_stage <- function(stage,data_root=file.path(.master_base(),'data'),output_root=file.path(.master_base(),'figures'),...){
  if(!stage %in% names(R_STAGES)) stop('Unknown stage: ',stage)
  dir.create(output_root,recursive=TRUE,showWarnings=FALSE)
  R_STAGES[[stage]](data_root=data_root,output_root=output_root,...)
}
.master_cli <- function(){
  args<-commandArgs(TRUE)
  val<-function(flag,default=NULL){i<-match(flag,args);if(is.na(i)) default else {if(i==length(args)) stop('Missing value: ',flag);args[i+1L]}}
  base<-.master_base()
  if('--list' %in% args || !length(args)){cat(paste(names(R_STAGES),collapse='\n'),'\n');return(invisible(NULL))}
  if('--check' %in% args){
    packages<-c('ggplot2','jsonlite','ragg','svglite','edgeR','Seurat','ggalluvial','plot1cell',
       'CellChat','ComplexHeatmap','slingshot','uwot','cowplot','pdftools','RColorBrewer','circlize','clustree','ggraph','ggrepel')
    report<-data.frame(package=packages,available=vapply(packages,requireNamespace,logical(1),quietly=TRUE))
    print(report,row.names=FALSE);return(invisible(report))
  }
  stage<-val('--stage')
  if(is.null(stage))stop('Use --list, --check, or --stage STAGE')
  data_root<-val('--data-root',file.path(base,'data'))
  output_root<-val('--output-root',file.path(base,'figures'))
  selected<-if(stage=='all') R_DEFAULT_STAGES else stage
  for(s in selected){
    message('Running ',s)
    if(s=='annotation_external_seurat') {
      src<-val('--seurat-path');if(is.null(src))stop('Provide --seurat-path for the actual external Seurat object')
      run_r_stage(s,data_root,output_root,seurat_path=src,pre_qc_path=val('--pre-qc-path'))
    } else run_r_stage(s,data_root,output_root)
  }
  capture.output(sessionInfo(),file=file.path(output_root,'R_session.txt'))
}
if(sys.nframe()==0L).master_cli()
