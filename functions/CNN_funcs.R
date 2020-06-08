# Functions to evaluate InceptionV3 and AlexNet results
# The functions assume strict rules for file structure
# Evaluation is done for separate (model saved based on each variables own accuracy)
# and common (model sasved based on the sum of variables' accuracies) cases

source("functions/keras_tf_funcs.R")

#' Evaluate results of separate models
#' 
#' @param methods Character, list of paths representing various methods for CNN training
#'        (file structure: method_path/predictions/[v,h,d]/files_with_predictions.csv)
#' @param obs.path Character, path to directory with observations
#' @param for.attrs Character, vector of forest attributes predicted by models,
#'        same as subfolders in "predictions" folder
#' @param d.sets Character, vector of separate dataset tags used during model training/testing,
#'        these have to be used in the naming of prediction/observation files, one needs to be "test"
#' @param out.name Character, name pattern (first part) of output file if NULL no files are exported
#' @param methods.bests Boolean, export the best results and relevant predictions for each method in methods
eval.separate <- function(methods,obs.path,for.attrs,d.sets,out.name=NULL,methods.bests=F) {
  # models were saved based on each forest attribute's accuracy separately
  # however predictions for all attributes were calculated
  # selecting best preforming predictions and accuracy scores
  cnn.bests <- lapply(methods,function(method) {
    cnn.f.a <- lapply(for.attrs,function(for.attr) {
      # input directory of prediction files
      in.dir <- paste0(method,"/predictions/",for.attr)
      # iterating through datasets
      cnn.d.set <- lapply(d.sets,function(d.set) {
        # list of prediction files
        in.files <- naturalsort::naturalsort(list.files(in.dir,d.set,full.names=T))
        # importing observed values
        obs.in <- list.files(obs.path,paste0(d.set,".csv"),full.names=T)
        # naming should be done unambiguously
        if (length(obs.in)>1) stop(paste0("multiple files match the pattern *",d.set,".csv in folder ",obs.path))
        # import observations of d.set
        obs <- read.csv(obs.in,as.is=T)
        # subset observations to include only forest attributes of interest
        obs <- obs[,for.attrs]
        # iterate through prediction files (created during parallel runs)
        cnn.results <- lapply(1:length(in.files),function(i) {
          # importing predictions of run #i
          cnn.res <- read.table(in.files[i],header=T,sep=";")
          # calculating RMSE and bias for run #i
          cnn.rmse <- rel.rmse(cnn.res,obs)
          cnn.bias <- rel.bias(cnn.res,obs)
          # returning accuracy scores and prediction table for e.g. scatter plots
          return(list(c(cnn.rmse,cnn.bias),cnn.res))
        })
        # extracting prediction tables for d.set (amount of runs times data frames with columns=for.attrs)
        cnn.preds <- lapply(cnn.results,"[[",2)
        names(cnn.preds) <- paste0(d.set,".",1:length(cnn.preds))
        # extracting accuracy scores for d.set (amount of runs times rows and 2xfor.attrs columns for RMSE and bias)
        cnn.results <- lapply(cnn.results,"[[",1)
        cnn.results <- as.data.frame(do.call(rbind,cnn.results))
        names(cnn.results) <- paste0(names(cnn.results),".",c(rep(paste0(d.set,".rmse"),3),rep(paste0(d.set,".bias"),3)))
        # returning accuracy scores and prediction table
        return(list(cnn.results,cnn.preds))
      })
      # extracting prediction tables for for.attr (for each element of d.sets amount of runs times data frames with columns=for.attrs)
      cnn.preds <- lapply(cnn.d.set,"[[",2)
      names(cnn.preds) <- d.sets
      # extracting accuracy scores for for.attr (amount of runs times rows and (2xfor.attrs)x(amount of d.sets) columns for RMSE and bias)
      cnn.d.set <- lapply(cnn.d.set,"[[",1)
      cnn.d.set <- do.call(cbind,cnn.d.set)
      return(list(cnn.d.set,cnn.preds))
    })
    # extracting prediction tables (for each element of for.attrs and for each element of d.sets amount of runs times data frames with columns=for.attrs)
    cnn.preds <- lapply(cnn.f.a,"[[",2)
    names(cnn.preds) <- paste0("sel.by.",for.attrs)
    # extracting accuracy scores (for each element of for.attrs amount of runs times rows and (2xfor.attrs)x(amount of d.sets) columns for RMSE and bias)
    cnn.f.a <- lapply(cnn.f.a,"[[",1)
    names(cnn.f.a) <- paste0("sel.by.",for.attrs)
    
    # now selecting the results for each method based on the best test run
    # other (e.g. traning, validation) results are selected from the same run as the best test result
    # saving original order of columns
    orig.names <- colnames(cnn.f.a[[1]])
    # selecting sel.by set based on minimum test RMSE of each forest attribute
    cnn.best <- lapply(1:length(for.attrs),function(i) {
      tmp <- lapply(cnn.f.a,function(x) {x[grep(paste0(for.attrs[i],".test.rmse"),names(x))]})
      tmp <- do.call(cbind,tmp)
      # indices of best RMSE (row (which run), column (which sel.by batch (one of v(=1),h(=2),d(=3))))
      sel.by <- which(tmp==min(tmp),arr.ind=T)
      # if multiple minimum values found get the one according to the sel.by batch if available else get first occurrence
      if (nrow(sel.by)>1) {if (any(sel.by[,2] %in% i)) sel.by <- sel.by[sel.by[,2] %in% i,] else sel.by[1,]}
      sel.by <- as.vector(sel.by)
      # extracting RMSEs and biases of for.attr according to best RMSE
      cnn.f.a.tmp <- cnn.f.a[[sel.by[2]]][grep(paste0("^",for.attrs[i],"."),colnames(cnn.f.a[[sel.by[2]]]))]
      # extracting predictions of for.attr according to best RMSE (sel.by batch (one of v,h,d), and run)
      cnn.pred.tmp <- lapply(cnn.preds[[sel.by[2]]],function(x) x[[sel.by[1]]][for.attrs[i]])
      return(list(cnn.f.a.tmp,cnn.pred.tmp))
    })
    # extracting predictions (for each element of d.sets predictions for each element of for.attrs)
    cnn.preds <- lapply(cnn.best,"[[",2)
    cnn.preds <- lapply(1:length(cnn.preds),function(i) dplyr::bind_cols(lapply(cnn.preds,"[[",i)))
    names(cnn.preds) <- d.sets
    # extracting accuracy scores for the method
    cnn.best <- lapply(cnn.best,"[[",1)
    cnn.best <- do.call(cbind,cnn.best)[orig.names]
    return(list(results=cnn.best,predictions=cnn.preds))
  })
  names(cnn.bests) <- basename(methods)
  
  # selecting the best method for each for.attr and relevant predictions
  orig.names <- colnames(cnn.bests[[1]]$results)
  # selecting method based on minimum test RMSE of each forest attribute
  cnn.best <- lapply(for.attrs,function(for.attr) {
    tmp <- lapply(cnn.bests,function(x) {x$results[grep(paste0(for.attr,".test.rmse"),names(x$results))]})
    tmp <- do.call(cbind,tmp)
    # selecting best method (in case of multiple the first is selected)
    sel.met <- which(tmp==min(tmp),arr.ind=T)[,2]
    if (length(sel.met)>1) {
      cat("More than one best method found, selecting first!\n")
    }
    # printing all best methods for for.attr
    dummy <- lapply(sel.met,function(i) cat(paste0("Best method(s) for ",for.attr,": ",names(cnn.bests)[i],"\n")))
    sel.met <- sel.met[1]
    cnn.bests.tmp <- cnn.bests[[sel.met]]$results[grep(paste0("^",for.attr,"."),colnames(cnn.bests[[sel.met]]$results))]
    cnn.pred.tmp <- lapply(cnn.bests[[sel.met]]$predictions,function(x) x[for.attr])
    return(list(cnn.bests.tmp,cnn.pred.tmp,names(cnn.bests)[sel.met]))
  })
  # names of best method for each attribute
  cnn.best.methods <- unlist(lapply(cnn.best,"[[",3))
  names(cnn.best.methods) <- for.attrs
  # summarizing resultsand exporting them (to file or returning from function)
  cnn.best.preds <- lapply(cnn.best,"[[",2)
  cnn.best.preds <- lapply(1:length(cnn.best.preds),function(i) dplyr::bind_cols(lapply(cnn.best.preds,"[[",i)))
  names(cnn.best.preds) <- d.sets
  cnn.best <- lapply(cnn.best,"[[",1)
  cnn.best <- do.call(cbind,cnn.best)[orig.names]
  # saving files with accuracy scores/predictions if name provided
  if (!is.null(out.name)) {
    saveRDS(cnn.best.preds,paste0(out.name,".preds.RDS"))
    saveRDS(cnn.best,paste0(out.name,".RDS"))
  }
  if (methods.bests) {
    return(list(best.rmse.bias=cnn.best,best.predictions=cnn.best.preds,best.methods=cnn.best.methods,methods.bests=cnn.bests))
  } else {
    return(list(best.rmse.bias=cnn.best,best.predictions=cnn.best.preds,best.methods=cnn.best.methods))
  }
}

#' Evaluate results of common models
#' 
#' @param methods Character, list of paths representing various methods for CNN training
#'        (file structure: method_path/predictions/files_with_predictions.csv)
#' @param obs.path Character, path to directory with observations
#' @param for.attrs Character, vector of forest attributes predicted by models,
#'        same as subfolders in "predictions" folder
#' @param d.sets Character, vector of separate dataset tags used during model training/testing,
#'        these have to be used in the naming of prediction/observation files, one needs to be "test"
#' @param out.name Character, name pattern (first part) of output file if NULL no files are exported
#' @param methods.res Boolean, export the results and predictions for each method in methods
eval.common <- function(methods,obs.path,for.attrs,d.sets,out.name=NULL,methods.res=F) {
  # models were saved based on forest attribute's summed accuracy
  # selecting best preforming predictions and accuracy scores
  cnn.methods <- lapply(methods,function(method) {
    # input directory of prediction files
    in.dir <- paste0(method,"/predictions")
    # iterating through datasets
    cnn.d.set <- lapply(d.sets,function(d.set) {
      # list of prediction files
      in.files <- naturalsort::naturalsort(list.files(in.dir,d.set,full.names=T))
      # importing observed values
      obs.in <- list.files(obs.path,paste0(d.set,".csv"),full.names=T)
      # naming should be done unambiguously
      if (length(obs.in)>1) stop(paste0("multiple files match the pattern *",d.set,".csv in folder ",obs.path))
      # import observations of d.set
      obs <- read.csv(obs.in,as.is=T)
      # subset observations to include only forest attributes of interest
      obs <- obs[,for.attrs]
      # iterate through prediction files (created during parallel runs)
      cnn.results <- lapply(1:length(in.files),function(i) {
        # importing predictions of run #i
        cnn.res <- read.table(in.files[i],header=T,sep=";")
        # calculating RMSE and bias for run #i
        cnn.rmse <- rel.rmse(cnn.res,obs)
        cnn.bias <- rel.bias(cnn.res,obs)
        # returning accuracy scores and prediction table for e.g. scatter plots
        return(list(c(cnn.rmse,cnn.bias),cnn.res))
      })
      # extracting prediction tables for d.set (amount of runs times data frames with columns=for.attrs)
      cnn.preds <- lapply(cnn.results,"[[",2)
      names(cnn.preds) <- paste0(d.set,".",1:length(cnn.preds))
      # extracting accuracy scores for d.set (amount of runs times rows and 2xfor.attrs columns for RMSE and bias)
      cnn.results <- lapply(cnn.results,"[[",1)
      cnn.results <- as.data.frame(do.call(rbind,cnn.results))
      names(cnn.results) <- paste0(names(cnn.results),".",c(rep(paste0(d.set,".rmse"),3),rep(paste0(d.set,".bias"),3)))
      # returning accuracy scores and prediction table
      return(list(cnn.results,cnn.preds))
    })
    # extracting prediction tables (for each element of d.sets amount of runs times data frames with columns=for.attrs)
    cnn.preds <- lapply(cnn.d.set,"[[",2)
    names(cnn.preds) <- d.sets
    # extracting accuracy scores (amount of runs times rows and (2xfor.attrs)x(amount of d.sets) columns for RMSE and bias)
    cnn.d.set <- lapply(cnn.d.set,"[[",1)
    cnn.d.set <- do.call(cbind,cnn.d.set)
    return(list(results=cnn.d.set,predictions=cnn.preds))
  })
  names(cnn.methods) <- basename(methods)
  
  # now selecting the results for each method based on the best test run
  # other (e.g. traning, validation) results are selected from the same run as the best test result
  # saving original order of columns
  orig.names <- colnames(cnn.methods[[1]]$results)
  # selecting method based on minimum test RMSE of each forest attribute
  cnn.best <- lapply(for.attrs,function(for.attr) {
    tmp <- lapply(cnn.methods,function(x) {x$results[grep(paste0(for.attr,".test.rmse"),names(x$results))]})
    tmp <- do.call(cbind,tmp)
    
    # indices of best RMSE (row (run), column (method)
    # if multiple minimum values found get the first one
    sel.met <- which(tmp==min(tmp),arr.ind=T)
    if (nrow(sel.met)>1) {
      cat("More than one best method found, selecting first!\n")
    }
    # printing all best methods for for.attr
    dummy <- lapply(sel.met[,2],function(i) cat(paste0("Best method(s) for ",for.attr,": ",names(cnn.methods)[i],"\n")))
    sel.met <- as.vector(sel.met[1,])
    cnn.bests.tmp <- cnn.methods[[sel.met[2]]]$results[grep(paste0("^",for.attr,"."),colnames(cnn.methods[[sel.met[2]]]$results))]
    cnn.pred.tmp <- lapply(cnn.methods[[sel.met[2]]]$predictions,function(x) x[[sel.met[1]]][for.attr])
    return(list(cnn.bests.tmp,cnn.pred.tmp,names(cnn.methods)[sel.met[2]]))
  })
  # names of best method for each attribute
  cnn.best.methods <- unlist(lapply(cnn.best,"[[",3))
  names(cnn.best.methods) <- for.attrs
  # summarizing resultsand exporting them (to file or returning from function)
  cnn.best.preds <- lapply(cnn.best,"[[",2)
  cnn.best.preds <- lapply(1:length(cnn.best.preds),function(i) dplyr::bind_cols(lapply(cnn.best.preds,"[[",i)))
  names(cnn.best.preds) <- d.sets
  cnn.best <- lapply(cnn.best,"[[",1)
  cnn.best <- do.call(cbind,cnn.best)[orig.names]
  # saving files with accuracy scores/predictions if name provided
  if (!is.null(out.name)) {
    saveRDS(cnn.best.preds,paste0(out.name,".preds.RDS"))
    saveRDS(cnn.best,paste0(out.name,".RDS"))
  }
  if (methods.res) {
    return(list(best.rmse.bias=cnn.best,best.predictions=cnn.best.preds,best.methods=cnn.best.methods,methods.res=cnn.methods))
  } else {
    return(list(best.rmse.bias=cnn.best,best.predictions=cnn.best.preds,best.methods=cnn.best.methods))
  }
}
  
