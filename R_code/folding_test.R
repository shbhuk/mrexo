n = 127
k.fold = 10
rand.gen <- sample(1:n, n) 
deg.length = 10
rand.gen <- sample(1:n, n)

for (i.degree in 2:3) {
  
  like.pred.vec <- rep(NA, k.fold)
  for (i.fold in 1:k.fold) {
    
    # creat indicator to separate dataset into training and testing datasets
    if (i.fold < k.fold) {
      split.interval <- ((i.fold-1)*floor(n/k.fold)+1):(i.fold*floor(n/k.fold))
    } else {
      split.interval <- ((i.fold-1)*floor(n/k.fold)+1):n
    }
    print(split.interval)
  }
}
    