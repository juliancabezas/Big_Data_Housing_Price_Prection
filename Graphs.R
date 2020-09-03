###################################
# Julian Cabezas Pena
# Big data Analysis and Project
# Project 1 1
# House price prediction using decision tree based methods
####################################


if (!require(tidyverse)) {install.packages("tidyverse")}

library(tidyverse)
library(gridExtra)
library(grid)

rfe_cat <- read.csv("./Intermediate_Results/RFE_CatBoost_gridscores.csv")
rfe_gbr <- read.csv("./Intermediate_Results/RFE_GBR_gridscores.csv")
rfe_xgbr <- read.csv("./Intermediate_Results/RFE_XGBR_gridscores.csv")
rfe_rf <- read.csv("./Intermediate_Results/RFE_RF_gridscores.csv")

rfe_cat$N_Variables <- 1:nrow(rfe_cat)
rfe_gbr$N_Variables <- 1:nrow(rfe_gbr)
rfe_xgbr$N_Variables <- 1:nrow(rfe_xgbr)
rfe_rf$N_Variables <- 1:nrow(rfe_rf)

rfe_cat$grid_scores <- sqrt(-rfe_cat$grid_scores)
rfe_gbr$grid_scores <- sqrt(-rfe_gbr$grid_scores)
rfe_xgbr$grid_scores <- sqrt(-rfe_xgbr$grid_scores)
rfe_rf$grid_scores <- sqrt(-rfe_rf$grid_scores)


catplot <- ggplot(data=rfe_cat, aes(x=N_Variables, y=grid_scores)) +
  geom_line(col="deepskyblue4")+
  geom_point(size=1,col="dodgerblue4") +
  annotate("point", x=which.min(rfe_cat$grid_scores),y=min(rfe_cat$grid_scores), colour = "red")+
  annotate("text",  x= max(rfe_cat$N_Variables), y =max(rfe_cat$grid_scores), label = "CatBoost", hjust=1) +
  ylab("RMSLE")+
  xlab("Number of Variables")+
  ylim(c(0.12,0.23)) +
  theme_bw(12)

catplot

which.min(rfe_cat$grid_scores)
min(rfe_cat$grid_scores)


gbrplot <- ggplot(data=rfe_gbr, aes(x=N_Variables, y=grid_scores)) +
  geom_line(col="deepskyblue4")+
  geom_point(size=1,col="dodgerblue4") +
  annotate("point", x=which.min(rfe_gbr$grid_scores),y=min(rfe_gbr$grid_scores), colour = "red")+
  annotate("text",  x= max(rfe_gbr$N_Variables), y =max(rfe_gbr$grid_scores), label = "GBR", hjust=1) +
  ylab("RMSLE")+
  xlab("Number of Variables")+
  ylim(c(0.12,0.23)) +
  theme_bw(12)

gbrplot

which.min(rfe_gbr$grid_scores)
min(rfe_gbr$grid_scores)


xgbrplot <- ggplot(data=rfe_xgbr, aes(x=N_Variables, y=grid_scores)) +
  geom_line(col="deepskyblue4")+
  geom_point(size=1,col="dodgerblue4") +
  annotate("point", x=which.min(rfe_xgbr$grid_scores),y=min(rfe_xgbr$grid_scores), colour = "red")+
  annotate("text",  x= max(rfe_xgbr$N_Variables), y =max(rfe_xgbr$grid_scores), label = "XGBR", hjust=1) +
  ylab("RMSLE")+
  xlab("Number of Variables")+
  ylim(c(0.12,0.23)) +
  theme_bw(12)

xgbrplot

which.min(rfe_xgbr$grid_scores)
min(rfe_xgbr$grid_scores)


rfplot <- ggplot(data=rfe_rf, aes(x=N_Variables, y=grid_scores)) +
  geom_line(col="deepskyblue4")+
  geom_point(size=1,col="dodgerblue4") +
  annotate("point", x=which.min(rfe_rf$grid_scores),y=min(rfe_rf$grid_scores), colour = "red")+
  annotate("text",  x= max(rfe_rf$N_Variables), y =max(rfe_rf$grid_scores), label = "RF", hjust=1) +
  ylab("RMSLE")+
  xlab("Number of Variables")+
  ylim(c(0.12,0.23)) +
  theme_bw(12)

rfplot

which.min(rfe_rf$grid_scores)
min(rfe_rf$grid_scores)


plots <- grid.arrange(rfplot,gbrplot,xgbrplot, catplot, ncol=2)

ggsave("./Document_latex/RFE.pdf",plots)




train <- read.csv("./Input_Data/train.csv")

nas<-sort(sapply(train, function(x) sum(is.na(x))),decreasing = T)

nas_df<-data.frame(Variable = factor(names(nas[1:18])), NAper = (nas[1:18]/nrow(train))*100)

nas_df

naplot <- ggplot(nas_df, aes(x = reorder(Variable, NAper), y = NAper))+
  geom_col(width = 0.9,fill = "dodgerblue4") +
  ylab("Percentage of NA values in the training data") +
  xlab("Variable") +
  coord_flip()+
  theme_bw(12)

naplot

ggsave("./Document_latex/NA_plot.pdf",naplot)




#Homemade function for histogram with boxplot
histoboxplot<-function (x,binw=round((max(x)-min(x))/30,1),xmin=floor(min(x)-(abs(min(x)*0.05))),
                        xmax=ceiling(max(x)+(max(x)*0.05)),ylab="Absolute Frequency",xlab="Variable",
                        title="histoboxplot",sumtable=FALSE,language="english") {
  # x = A vector
  # lenguage can be "spanish or "english"
  
  # Convert x to data.frame
  df<-data.frame(x=x)
  
  #Calculate frecuencies according to binwhith
  df$freq<-cut(df$x,breaks=seq(xmin,xmax,binw))
  
  #Calculate maximum frequency
  ymax<-max(tapply(df$x,df$freq,length),na.rm=TRUE)
  
  #More or less 30% of extra space for the boxplot
  ymax<-ymax+ymax*0.3
  
  # Histogram
  hist_in<-ggplot(df, aes(x=x)) +
    geom_histogram(color="black",fill="dodgerblue4",breaks=seq(xmin,xmax,binw),size=0.2) +
    ylab(ylab) +
    xlim(c(xmin,xmax)) +
    #xlim(c(-99,10)) +
    ylim(c(0,ymax)) +
    xlab(xlab) +
    ggtitle(title) +
    theme_bw(12)
  
  # Boxplot
  box_in<-ggplot(df,aes(x=factor(0),x))+
    geom_boxplot(outlier.size = 0.8,lwd=0.5,fill="dodgerblue4")+
    coord_flip() +
    scale_y_continuous(limits=c(xmin, xmax), expand = c(0, 0)) +
    theme(axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank(),
          axis.title.y=element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank(),
          plot.margin=unit(c(0, 0, 0, 0), "null"),
          axis.text = element_blank(), axis.ticks.length = unit(0, "mm"),
          #panel.background = element_rect(fill = "red"), # bg of the panel
          #plot.background = element_rect(fill = "transparent", col = NA), # bg of the plot
          #panel.grid.major = element_blank(), # get rid of major grid
          #panel.grid.minor = element_blank(), # get rid of minor grid
          legend.background = element_rect(fill = "transparent"), # get rid of legend bg
          legend.box.background = element_rect(fill = "transparent"),
          panel.border = element_rect(colour = "black", fill=NA, size=0.5)
    ) +
    labs(x = NULL, y = NULL)
  
  #Cobvert to grob object
  box_grob <- ggplotGrob(box_in)
  
  # Insert the boxplot in the histogram
  hist_box<-hist_in + 
    annotation_custom(grob = box_grob, xmin = xmin, xmax = xmax, 
                      ymin = ymax*0.85, ymax = ymax)
  
  # In case you want a summary table
  if (sumtable) {
    
    #Summary table
    sum<-summary(df$x)
    
    # Calculate CV
    sum[7]<-(sd(df$x)/mean(df$x))*100
    
    if (language=="spanish") {
      sum.df<-data.frame(cbind(Estadístico=c("Mínimo","Mediana","Media","Máximo","CV (%)"),Valor=round(as.vector(sum),2)[c(1,3,4,6,7)]),row.names=NULL)
    }
    
    if (language=="english") {
      sum.df<-data.frame(cbind(Statistic =c("Minimum","Median","Mean","Maximum","CV (%)"),Value=round(as.vector(sum),2)[c(1,3,4,6,7)]),row.names=NULL)
    }
    
    if (!(language %in% c('spanish','english'))) {
      stop("languaje not supported, select spanish or english")
    }
    
    # Set theme for the table
    tt1 <- tt1 <- gridExtra::ttheme_default(
      core = list(fg_params=list(cex = 1)),
      colhead = list(fg_params=list(cex = 1)),
      rowhead = list(fg_params=list(cex = 1)))
    
    sum.grob<-tableGrob(sum.df, theme=tt1,rows = NULL)
    
    # Insertar la dataframe en el histograma
    histoboxplot_full<-hist_box + 
      annotation_custom(grob = sum.grob, xmin = xmax*0.6, xmax = xmax, 
                        ymin = ymax*0.4, ymax = ymax*0.6)
    
    return(histoboxplot_full)
    
  } else {
    return(hist_box)
  }
}

options(scipen=999)
hist_raw=histoboxplot(train$SalePrice,title="",xlab = "House Sale Price ($US)")
ggsave("./Document_latex/saleprice_hist.pdf",hist_raw)


hist_log=histoboxplot(log(train$SalePrice+1),title="",xlab = "Log(House Sale Price  + 1)")
ggsave("./Document_latex/saleprice_log.pdf",hist_log)

grid<-grid.arrange(hist_raw,hist_log,ncol=2)
ggsave("./Document_latex/2hist_saleprice.pdf",grid)


