rm(list=ls())

# packages <- c("tidyverse", "data.table", "ggplot2", "ggpubr")


# # Répertoire local pour les packages
# local_lib <- "~/Rlibs"

# # Créer le répertoire si besoin
# if (!dir.exists(local_lib)) {
#   dir.create(local_lib, recursive = TRUE)
# }

# # Ajouter à la liste des chemins
# .libPaths(c(local_lib, .libPaths()))

# # Fonction pour installer les packages manquants
# install_if_missing <- function(pkg) {
#   if (!require(pkg, character.only = TRUE)) {
#     message(paste("Installation du package :", pkg))
#     install.packages(pkg, lib = local_lib, dependencies = TRUE)
#     library(pkg, character.only = TRUE)
#   } else {
#     message(paste("Package déjà installé :", pkg))
#   }
# }

# # Appliquer à tous les packages
# invisible(lapply(packages, install_if_missing))

library(tidyverse)
library(data.table)
library(ggplot2)
library(ggpubr)

root_dir <- "/rex/local/otmedia/medialex"
plot_dir <- file.path(root_dir, "plots")

# sim_file_01 <- file.path(root_dir, "nahel_sim_wiki01.csv")
# sim_file_02 <- file.path(root_dir, "nahel_sim_wiki02.csv")
# prefix <- "wiki"

sim_file <- file.path(root_dir, "matrix/nahel_sim_wiki.csv")
# sim_file_01 <- file.path(root_dir, "titan_sim_wiki01.csv")
# sim_file_02 <- file.path(root_dir, "titan_sim_wiki02.csv")
prefix <- "nahel"

theme_nrv <- theme_bw(14) +
  theme(
    axis.text = element_text(size = rel(1)),
    axis.title = element_text(size = rel(1)),
    strip.text = element_text(size = rel(1)),
    strip.background = element_rect(fill = "grey90"),
    legend.text = element_text(size = rel(1))
  )

setwd(root_dir)

raw_data <- as.data.frame(fread(file=sim_file, check.names=FALSE, head=TRUE, stringsAsFactors=FALSE, sep=","))
# raw_data_01 <- as.data.frame(fread(file=sim_file_01, check.names=FALSE, head=TRUE, stringsAsFactors=FALSE, sep=","))
# raw_data_02 <- as.data.frame(fread(file=sim_file_02, check.names=FALSE, head=TRUE, stringsAsFactors=FALSE, sep=","))
# raw_data <- rbind(raw_data_01, raw_data_02)


max(raw_data$max)

data_to_plot <- raw_data %>% filter(channel =="BFM_TV")
myplot <- ggplot(xaxs="i", yaxs="i")
myplot <- myplot + geom_line(data=data_to_plot, aes(x=start, y=max))
myplot <- myplot + scale_x_datetime() + ylim(0, 0.8)
myplot <- myplot + theme_nrv + labs(x="", y="max", title = "BFM_TV")
myplot

for (a_single_channel in unique(raw_data$channel)) {
  data_to_plot <- raw_data %>% filter(channel == a_single_channel)
  myplot <- ggplot(xaxs="i", yaxs="i")
  myplot <- myplot + geom_line(data=data_to_plot, aes(x=start, y=max))
  myplot <- myplot + scale_x_datetime()
  myplot <- myplot + theme_nrv + labs(x="", y="max", title = a_single_channel) + ylim(0, 0.8)
  finalFilename <- paste("_01__per_channel_", a_single_channel, ".pdf", sep="")
  finalFilename <- gsub('ç', 'c', finalFilename)
  finalFilename <- gsub(' ', '_', finalFilename)
  finalFilename <- gsub(':', '', finalFilename)
  ggexport(myplot, filename = file.path(plot_dir, paste(prefix, "_", finalFilename, sep="")), width = 19, height = 10)
}



data_to_plot <- raw_data %>% filter(channel =="BFM_TV") %>% filter(start > as.POSIXct("2023-06-18 05:00:00") & start < as.POSIXct("2023-06-19 01:00:00"))
myplot <- ggplot(xaxs="i", yaxs="i")
myplot <- myplot + geom_line(data=data_to_plot, aes(x=start, y=max))
myplot <- myplot + scale_x_datetime(date_labels = "%H:%M", breaks = scales::date_breaks("60 mins"))
myplot <- myplot + theme_nrv + labs(x="", y="max", title = "BFM_TV")
myplot

for (a_single_channel in unique(raw_data$channel)) {
  data_to_plot <- raw_data %>% filter(channel ==a_single_channel) %>% filter(start > as.POSIXct("2023-06-18 05:00:00") & start < as.POSIXct("2023-06-19 01:00:00"))
  myplot <- ggplot(xaxs="i", yaxs="i")
  myplot <- myplot + geom_line(data=data_to_plot, aes(x=start, y=max))
  myplot <- myplot + scale_x_datetime(date_labels = "%H:%M", breaks = scales::date_breaks("60 mins"))
  myplot <- myplot + theme_nrv + labs(x="", y="max", title = a_single_channel)+ ylim(0, 0.8)
  finalFilename <- paste("_02__per_channel_", a_single_channel, ".pdf", sep="")
  finalFilename <- gsub('ç', 'c', finalFilename)
  finalFilename <- gsub(' ', '_', finalFilename)
  finalFilename <- gsub(':', '', finalFilename)
  ggexport(myplot, filename = file.path(plot_dir, paste(prefix, "_", finalFilename, sep="")), width = 19, height = 10)
}



data_to_plot <- raw_data %>% filter(channel %in% c("TF1", "France2")) %>% filter(start > as.POSIXct("2023-06-27 01:00:00") & start < as.POSIXct("2023-06-28 01:00:00"))
myplot <- ggplot(xaxs="i", yaxs="i")
myplot <- myplot + geom_line(data=data_to_plot, aes(x=start, y=max, color=channel))
myplot <- myplot + scale_x_datetime(date_labels = "%H:%M", breaks = scales::date_breaks("60 mins"))
myplot <- myplot + theme_nrv + labs(x="", y="max") + theme(legend.position = "top")
myplot

for (day in c("06-27", "06-28", "06-29", "06-30", "07-01", "07-02", "07-03")) {
# for (day in c("06-18", "06-19", "06-20", "06-21", "06-22", "06-23", "06-24", "06-25", "06-26", "06-27", "06-28", "06-29", "06-30")) {  
  print(paste("2023-", day, sep=""))
  data_to_plot <- raw_data %>% filter(channel %in% c("TF1", "France2")) %>% filter(start > as.POSIXct(paste("2023-", day, " 04:00:00", sep="")) & start < as.POSIXct(paste("2023-", day, " 23:59:59", sep="")))
  myplot <- ggplot(xaxs="i", yaxs="i")
  myplot <- myplot + geom_line(data=data_to_plot, aes(x=start, y=max, color=channel))
  myplot <- myplot + scale_x_datetime(date_labels = "%H:%M", breaks = scales::date_breaks("60 mins")) + ylim(0, 0.8)
  myplot <- myplot + theme_nrv + labs(x="", y="max", color=paste("2023-", day, sep="")) + theme(legend.position = "top")
  finalFilename <- paste("_03__tf1_fr2_", day, ".pdf", sep="")
  ggexport(myplot, filename = file.path(plot_dir, paste(prefix, "_", finalFilename, sep="")), width = 19, height = 10)
}

for (day in c("06-27", "06-28", "06-29", "06-30", "07-01", "07-02", "07-03")) {
# for (day in c("06-18", "06-19", "06-20", "06-21", "06-22", "06-23", "06-24", "06-25", "06-26", "06-27", "06-28", "06-29", "06-30")) { 
  print(paste("2023-", day, sep=""))
  data_to_plot <- raw_data %>% filter(channel %in% c("BFM_TV", "CNews", "LCI", "FranceInfo_TV")) %>% filter(start > as.POSIXct(paste("2023-", day, " 04:00:00", sep="")) & start < as.POSIXct(paste("2023-", day, " 23:59:59", sep="")))
  myplot <- ggplot(xaxs="i", yaxs="i")
  myplot <- myplot + geom_line(data=data_to_plot, aes(x=start, y=max, color=channel))
  myplot <- myplot + scale_x_datetime(date_labels = "%H:%M", breaks = scales::date_breaks("60 mins")) + ylim(0, 0.8)
  myplot <- myplot + theme_nrv + labs(x="", y="max", color=paste("2023-", day, sep="")) + theme(legend.position = "top")
  finalFilename <- paste("_04__chaines_info_", day, ".pdf", sep="")
  ggexport(myplot, filename = file.path(plot_dir, paste(prefix, "_", finalFilename, sep="")), width = 19, height = 10)
}


data_to_plot <- raw_data %>% filter(channel %in% c("TF1", "France2")) %>% filter(start > as.POSIXct("2023-06-27 01:00:00") & start < as.POSIXct("2023-06-28 01:00:00"))
myplot <- ggplot(xaxs="i", yaxs="i")
myplot <- myplot + geom_line(data=data_to_plot, aes(x=start, y=max, color=channel))
myplot <- myplot + scale_x_datetime(date_labels = "%H:%M", breaks = scales::date_breaks("60 mins"))
myplot <- myplot + theme_nrv + labs(x="", y="max") + theme(legend.position = "top")
myplot
finalFilename <- paste("_03__tf1_fr2_27", ".pdf", sep="")
ggexport(myplot, filename = file.path(plot_dir, paste(prefix, "_", finalFilename, sep="")), width = 19, height = 10)



data_to_plot <- raw_data %>% filter(channel %in% c("BFM_TV", "CNews", "LCI", "FranceInfo_TV")) %>% filter(start > as.POSIXct("2023-06-27 03:00:00") & start < as.POSIXct("2023-06-28 01:00:00"))
myplot <- ggplot(xaxs="i", yaxs="i")
myplot <- myplot + geom_line(data=data_to_plot, aes(x=start, y=max, color=channel))
myplot <- myplot + scale_x_datetime(date_labels = "%H:%M", breaks = scales::date_breaks("30 mins"))
myplot <- myplot + theme_nrv + labs(x="", y="max")
myplot



