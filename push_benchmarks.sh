#!/bin/bash
git checkout $BRANCH_NAME --

if [ "$?" -ne "0" ] ; then
    LOCAL_BRANCH_NAME="temp_bmark"
    git fetch origin pull/$pullrequest/head:$LOCAL_BRANCH_NAME
    git checkout $LOCAL_BRANCH_NAME --
fi
julia benchmark/send_comment_to_pr.jl -o $org -r $repo -p $pullrequest -c '**Starting benchmarks!**'

julia benchmark/$1 $repo

if [ "$?" -eq "0" ] ; then
    julia benchmark/send_comment_to_pr.jl -o $org -r $repo -p $pullrequest -g
else
    julia benchmark/send_comment_to_pr.jl -o $org -r $repo -p $pullrequest -c "**An error has occured while running the benchmark script: $1** "
fi

git checkout master

git branch -D $LOCAL_BRANCH_NAME
