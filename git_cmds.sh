###### Dibakar's Bash Setup ######

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

### Prompt

grey='\[\033[1;30m\]'
red='\[\033[0;31m\]'
RED='\[\033[1;31m\]'
green='\[\033[0;32m\]'
GREEN='\[\033[1;32m\]'
yellow='\[\033[0;33m\]'
YELLOW='\[\033[1;33m\]'
purple='\[\033[0;35m\]'
PURPLE='\[\033[1;35m\]'
white='\[\033[0;37m\]'
WHITE='\[\033[1;37m\]'
blue='\[\033[0;34m\]'
BLUE='\[\033[1;34m\]'
cyan='\[\033[0;36m\]'
CYAN='\[\033[1;36m\]'
NC='\[\033[0m\]'
PS1='\[\033[1;34m\]$USER@\h:\W\>\[\e[m\] '

c() {
    code "$1" &
}
