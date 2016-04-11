#!/usr/bin/perl
#
#  Author: Preslav Nakov
#  
#  Description: Scores SemEval-2014 task 9, subtask B
#
#  Last modified: April 7, 2014
#
#

use warnings;
use strict;
use utf8;
binmode (STDIN,  ":utf8");
binmode (STDOUT, ":utf8");

my $INPUT_FILE         =  $ARGV[0];
use constant GOLD_FILE => 'eval/semeval-tweet-test-B-reference.txt';
my $OUTPUT_FILE        =  $INPUT_FILE . '.scored';


########################
###   MAIN PROGRAM   ###
########################

my %stats = ();

### 1. Read the files and get the statsitics
open INPUT, '<:encoding(UTF-8)', $INPUT_FILE or die;
open GOLD,  '<:encoding(UTF-8)', GOLD_FILE or die;

my $lineNo = 1;
for (; <INPUT>; $lineNo++) {
	s/^[ \t]+//;
	s/[ \t\n\r]+$//;

	### 1.1. Check the input file format
	#NA	1	positive	i'm done writing code for the week! Looks like we've developed a good a** game for the show Revenge on ABC Sunday, Premeres 09/30/12 9pm
	die "Wrong file format!" if (!/^NA\t$lineNo\t(positive|negative|neutral)/);
	my $proposedLabel = $1;

	### 1.2	. Check the gold file format
	#NA	T14114531	positive
	$_ = <GOLD>;
	die "Wrong file format!" if (!/^NA\t([^\t]+)\t(positive|negative|neutral)/);
	my ($dataset, $trueLabel) = ($1, $2);

	### 1.3. Update the statistics
	$stats{$dataset}{$proposedLabel}{$trueLabel}++;
}

close(INPUT) or die;
close(GOLD) or die;
$lineNo--;
die "Too few lines: $lineNo" if (8987 != $lineNo);

### 2. Initialize zero counts
foreach my $dataset (keys %stats) {
	foreach my $class1 ('positive', 'negative', 'neutral') {
		foreach my $class2 ('positive', 'negative', 'neutral') {
			$stats{$dataset}{$class1}{$class2} = 0 if (!defined($stats{$dataset}{$class1}{$class2}))
		}
	}
}

### 3. Calculate the F1 for each dataset
print "$INPUT_FILE\t";
open OUTPUT, '>:encoding(UTF-8)', $OUTPUT_FILE or die;
foreach my $dataset (sort keys %stats) {
	print OUTPUT "\nScoring $dataset:\n";
	print "$dataset\t";

	my $overall = 0.0;
	foreach my $class ('positive', 'negative', 'neutral') {
		my $denomP = ($stats{$dataset}{'positive'}{$class} + $stats{$dataset}{'negative'}{$class} + $stats{$dataset}{'neutral'}{$class}) > 0 ? ($stats{$dataset}{'positive'}{$class} + $stats{$dataset}{'negative'}{$class} + $stats{$dataset}{'neutral'}{$class}) : 1;
		my $P = 100.0 * $stats{$dataset}{$class}{$class} / $denomP;

		my $denomR = (($stats{$dataset}{$class}{'positive'} + $stats{$dataset}{$class}{'negative'} + $stats{$dataset}{$class}{'neutral'}) > 0) ? ($stats{$dataset}{$class}{'positive'} + $stats{$dataset}{$class}{'negative'} + $stats{$dataset}{$class}{'neutral'}) : 1;
		my $R = 100.0 * $stats{$dataset}{$class}{$class} / $denomR;
		
		my $denom = ($P+$R > 0) ? ($P+$R) : 1;
		my $F1 = 2*$P*$R / $denom;

		$overall += $F1 if ($class ne 'neutral');
		printf OUTPUT "\t%8s: P=%0.2f, R=%0.2f, F1=%0.2f\n", $class, $P, $R, $F1;
	}
	$overall /= 2.0;
	printf OUTPUT "\tOVERALL SCORE : %0.2f\n", $overall;
	printf "%0.2f\t", $overall;
}
print "\n";
close(OUTPUT) or die;
