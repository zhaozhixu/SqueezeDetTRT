#! /usr/bin/perl

use warnings;
use strict;

my $usage ="usage: $0 -t TYPE -o DEST SOURCE(s)\n" .
'Convert input tensor file SOURCE(s) to one output weight file DEST to be used by TensorRT.

[options]
  -t, --type	weights data type, allowed 0 (float), 1 (half), 2 (int8), currently only support 0
  -o, --out	followed by output file name

Author: Zhao Zhixu
';
my $ofname = "out.wts";
my $type = "0";
if (@ARGV == 0) {
  print $usage;
  exit;
}
while (my $opt = shift @ARGV) {
  if ($opt eq '-h' || $opt eq '--help')  {
    print $usage;
    exit;
  }
  elsif ($opt eq '-t' || $opt eq '--type') {
    $type = shift @ARGV;
  }
  elsif ($opt eq '-o' || $opt eq '--out') {
    $ofname = shift @ARGV;
    last;
  }
  else {
    print $usage;
    exit;
  }
}

open OUTFILE, ">$ofname" or die "Can't open file ${ofname}. ($!)";
my $nweight = @ARGV;
print OUTFILE "$nweight\n";

while (my $fname = shift @ARGV) {
  next if -d $fname;
  open INFILE, "<$fname" or die "Can't open file ${fname}. ($!)";
  print "Processing file ${fname}...";

  my @floats;
  my @outputs;
  while (<INFILE>) {
    my @words = split /\s+/, $_;
    foreach (@words) {
      push @floats, $1 if (/([-+]?\d+\.?(\d*e?[-+]?\d+)?)/);
    }
  }
  my $nfloat = @floats;
  (my $wname = $fname) =~ s/:0.*//;
  $wname =~ s|.*/||;
  print OUTFILE "$wname $type $nfloat";
  foreach (@floats) {
    push @outputs, join "", reverse split /(?=[0-9a-fA-F])/, unpack("h*", pack("f*", $_));
  }
  print OUTFILE " " . (join " ", @outputs) . "\n";
  print "done\n";
  close INFILE;
}

close OUTFILE;
