library(data.table)


# List all filenames
filenames <- c(
  list.files(path = "dataset/train", full.names = TRUE),
  list.files(path = "dataset/test", full.names = TRUE)
)


# All dataset
all_data <- as.data.table(read.csv("dataset/all_dataset.csv"))[, -"indexes"]

# Clinical data
clinical <- as.data.table(read.csv("dataset/clinical.csv"))[,
  c("case_submitter_id", "age_at_index", "ethnicity", "gender", "race")]
clinical <- clinical[!duplicated(clinical$case_submitter_id), ]
colnames(clinical)[1] <- "TCGA.ID"

res <- merge(clinical, all_data, by = "TCGA.ID")
res$age_at_index <- as.numeric(res$age_at_index)
colnames(res)[2] <- "Age (at index)"


# Crate data.table
df <- data.table(
  TCGA.ID = gsub("dataset/train/|dataset/test/|.png", "", filenames) |>
    strsplit("-") |>
    sapply(\(x) paste0(x[1:3], collapse = "-")),
  train = grepl("train", filenames),
  full_path = gsub("dataset/train/|dataset/test/", "", filenames)
)

res <- merge(res, df, by = "TCGA.ID")

write.csv(res, "dataset/all_data_custom.csv", row.names = FALSE)

