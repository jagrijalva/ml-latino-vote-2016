# Check what C2, C3, C4, C5, C8, C9 measure
load("CMPS_2016_raw.rda")
df <- da38040.0001

# Filter to Latinos
latinos <- df[df$ETHNIC_QUOTA == "(2) Hispanic or Latino", ]

vars_to_check <- c("C2", "C3", "C4", "C5", "C8", "C9", "C10", "C11",
                   "C25", "C31", "C38", "C40", "C41", "C45",
                   "C111", "C114", "C115", "C142", "C158",
                   "C228", "C247", "C248", "C337",
                   "L46", "L266", "L267", "L195_1", "L195_2", "L195_3",
                   "BL155", "BL229", "BLA206", "BLA207")

for (v in vars_to_check) {
  if (v %in% names(latinos)) {
    cat("\n", strrep("=", 60), "\n")
    cat("Variable:", v, "\n")
    cat(strrep("-", 40), "\n")
    print(table(latinos[[v]], useNA = "ifany"))
  }
}
