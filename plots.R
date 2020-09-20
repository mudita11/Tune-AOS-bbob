library(ggplot2)
# Compute percentages with dplyr
library(dplyr)

pdf_crop <- function(filename)
{
  system2("pdfcrop", c("--pdfversion 1.5", filename, filename))
  # This has to go after pdfcrop
  embedFonts(filename)
}

plot_op_cumfreq <- function(filename, pdf_file = NULL, pdf_scale = 1, pdf_width = 6, pdf_height = 3)
{
  data <- read.table(filename)[,1:2]
  colnames(data) <- c("op", "Generation")
  data$op <- factor(data$op, labels=c("rand/1", "rand/2", "rand-to-best/2", "current-to-rand/1", "current_to_pbest", "current_to_pbest_archived", "best/1", "current_to_best/1", "best/2"))
  
  max_gen <- max(data$Generation)
  popsize <- sum(data$Generation == 0)
  
  data <- data  %>%
    group_by(Generation, op) %>%
    summarise(Applications = length(op)) %>%
    ungroup() %>% group_by(op) %>%
    mutate(cum_appl = cumsum(Applications)) %>%
    ungroup() %>%   group_by(Generation) %>%
    mutate(cum_appl_frac = 100 * cum_appl / sum(cum_appl)) %>%
    ungroup()
  
  gg <- ggplot(data, aes(x=Generation, y=cum_appl_frac, fill=op)) + 
    geom_area(stat = "identity", position = "stack") + # size=0.5, colour=NA) +
    scale_x_continuous(limits = c(0, max_gen), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 101), expand = c(0, 0)) +
    scale_fill_viridis_d() +
    ylab("Cumulative percentage of applications") +
    theme_bw() +
    theme(panel.grid = element_blank(),
          panel.border = element_blank(),
          legend.title = element_blank(),
          legend.text=element_text(size=7))
  if (!is.null(pdf_file)) {
    ggsave(gg, filename=pdf_file,
           width = pdf_scale * pdf_width, height = pdf_scale * pdf_height)
    pdf_crop(pdf_file)
  }
  print(gg)
  invisible(gg)
}


for (file in list.files("exdata/Hybrid", "genw_.*.gz", full.names=TRUE)) {
  pdf_file <- sub("txt.gz", "pdf",
                  sub("genw", "opfreq",
                      sub("exdata", "figs", file, fixed=TRUE), fixed=TRUE), fixed=TRUE)
  gg <- plot_op_cumfreq(file, pdf_file = pdf_file)
}


