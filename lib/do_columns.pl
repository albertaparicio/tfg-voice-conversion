#!/usr/bin/perl -w
use Getopt::Std;

$opt_c=10;
$opt_s = "[ \t,]+";
$opt_h = 0;

sub syntax {
     print STDERR "Usage: $0 [-c numcol] [-s sep] [files]\n\n";
     print STDERR "\t-c        numcol (def. $opt_c\n";
     print STDERR "\t-s        input_separator (def. $opt_s\n";

}

syntax, exit 1 if (!getopts('c:s:h') || $opt_h);




$n = 0;
while (<>) {
    chomp;
    @F = split /$opt_s/;
    foreach $d (@F) {
	print "\t" if $n>0;
	print $d;
	if (++$n == $opt_c) {print "\n"; $n=0}
    }
}
if ($n != 0) {print "\n"}
exit 0;
