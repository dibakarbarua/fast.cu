git config --global core.editor "vim"
git config --global user.email "dibakar.barua92@gmail.com"
git config --global user.name "Dibakar Barua"

export EDITOR=vim
alias ls="ls -h --color"
alias ll="ls -l"
alias vi="vim"
alias h="history"
alias h="history"
alias d="dirs"
alias a="alias"
alias l="ls -a -C -F"
alias lc="ls -a -l -F | less"
alias ll="ls -altr"
alias lls="ls -altr | less"
alias lh="ls -altrh"
alias j="jobs -l"
alias m="more -c"
alias cls="clear"
alias bye="clear; exit"
alias pg="pg -ne"
alias pspg="ps -ef | less"
alias psg="ps -ef | grep"
alias r="rsh"
alias vi="vim"
alias gdb='gdb -q'
alias rmlogs='rm *csv *log *td'

c() {
    code "$1" &
}
