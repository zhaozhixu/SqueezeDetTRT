#! /usr/bin/perl

use warnings;
use strict;

my $usage = <<EOF;
Usage: $0 IOU_FILE IOU
Print statistic about datasets whose ious are below IOU.
EOF
if (@ARGV < 2 or @ARGV > 2) {
  print $usage;
  exit;
}

my $iou_file = $ARGV[0];
my $iou_level = $ARGV[1];

my $dataset;
my $iou;
my %count;
open IOU_FILE, '<', $iou_file or die "Cannot open $iou_file: $!";
while (<IOU_FILE>) {
  if (/(\d+)\d{4}\.txt: (\d(\.\d*)?)/) {
    $dataset = $1;
    $iou = $2;
    next if $iou > $iou_level;
    $count{$dataset} = 0 if not exists $count{$dataset};
    $count{$dataset}++;
  }
}
close IOU_FILE;

my $bar_width = 50;
my $max_count = (sort {$a <=> $b} values %count)[-1];
my @count_names = reverse sort {$count{$a} <=> $count{$b}} keys %count;

foreach (@count_names) {
  printf "%3s: ", $_;
  print '*'x($count{$_}/$max_count*$bar_width)." $count{$_}\n";
}

my $count_num = keys %count;
my $record_num = &sum(values %count);
print "$count_num datasets, $record_num records\n";

sub sum {
  my $n = 0;
  foreach (@_) {
    $n += $_;
  }
  $n;
}
