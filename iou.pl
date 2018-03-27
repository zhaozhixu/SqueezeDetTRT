#! /usr/bin/perl

use strict;
use warnings;
use File::Spec::Functions;

my $usage = <<EOF;
Usage: $0 XML_DIR LABEL_DIR
Compute the iou of bounding boxes in XML_DIR and LABEL_DIR.
EOF
if (@ARGV < 2) {
  print $usage;
  exit;
}

my $xml_dir = $ARGV[0];
my $label_dir = $ARGV[1];
my @xml_files = glob "$xml_dir/*.xml";
my $xml_num = @xml_files;
my $count = 0;
my $bar_width = 70;
my @ious;

opendir XML_DIR, $xml_dir or die "Cannot open $xml_dir: $!";
foreach (readdir XML_DIR) {
  next unless /\.xml$/;
  my $xml = catfile($xml_dir, $_);
  s/\.xml/.txt/;
  my $label = catfile($label_dir, $_);
  open XML, '<', $xml or die "Cannot open $xml: $!";
  my $xml_str = join "", <XML>;
  my $xmin = $1 if $xml_str =~ /<xmin>(.+)<\/xmin>/;
  my $ymin = $1 if $xml_str =~ /<ymin>(.+)<\/ymin>/;
  my $xmax = $1 if $xml_str =~ /<xmax>(.+)<\/xmax>/;
  my $ymax = $1 if $xml_str =~ /<ymax>(.+)<\/ymax>/;
  close XML;
  open LABEL, '<', $label or die "Cannot open $label: $!";
  my $label_str = join "", <LABEL>;
  my ($xmin_gt, $ymin_gt, $xmax_gt, $ymax_gt) = (split " ", $label_str)[4, 5, 6, 7];
  close LABEL;

  my $xmin_inter = &max($xmin, $xmin_gt);
  my $ymin_inter = &max($ymin, $ymin_gt);
  my $xmax_inter = &min($xmax, $xmax_gt);
  my $ymax_inter = &min($ymax, $ymax_gt);
  my $area_inter = &area($xmin_inter, $ymin_inter, $xmax_inter, $ymax_inter);
  my $area_union = &area($xmin, $ymin, $xmax, $ymax) +
    &area($xmin_gt, $ymin_gt, $xmax_gt, $ymax_gt) - $area_inter;
  my $iou = $area_inter / $area_union;
  push @ious, $iou;

  my $bar_count = ++$count / $xml_num * $bar_width;
  $| = 1;
  my $percent = sprintf("%.1f", $bar_count/$bar_width*100);
  print "\r[".("#"x$bar_count).(" "x($bar_width-$bar_count))."]".$percent."% ";
  # print "\n";
  # print "xml: $xml, label: $label\n";
  # print "xmin = $xmin, ymin = $ymin, xmax = $xmax, ymax = $ymax\n";
  # print "xmin_gt = $xmin_gt, ymin_gt = $ymin_gt, xmax_gt = $xmax_gt, ymax_gt = $ymax_gt\n";
  # print "xmin_inter = $xmin_inter, ymin_inter = $ymin_inter, xmax_inter = $xmax_inter, ymax_inter = $ymax_inter\n";
  # print "area_inter = $area_inter, area_union = $area_union\n";
  # print "iou = $iou\n";
  # exit;
}
print "\n";
closedir XML_DIR;

my $mean_iou = &sum(@ious) / @ious;
my $n_iou = @ious;
print "Total $n_iou records\n";
print "Mean IoU: $mean_iou\n";

sub area {
  my ($xmin, $ymin, $xmax, $ymax) = @_;
  my $a;
  if ($xmax < $xmin or $ymax < $ymin) {
    $a = 0;
  } else {
    $a = ($xmax - $xmin) * ($ymax - $ymin);
  }
  $a;
}

sub max {
  my $ret;
  if ($_[0] > $_[1]) {
    $ret = $_[0];
  } else {
    $ret = $_[1];
  }
  $ret;
}

sub min {
  my $ret;
  if ($_[0] < $_[1]) {
    $ret = $_[0];
  } else {
    $ret = $_[1];
  }
  $ret;
}

sub sum {
  my $s = 0;
  foreach (@_) {
    $s += $_;
  }
  $s;
}
