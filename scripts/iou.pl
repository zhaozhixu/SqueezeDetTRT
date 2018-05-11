#! /usr/bin/perl

use strict;
use warnings;
use File::Spec::Functions;

my $usage = <<EOF;
Usage: $0 XML_DIR LABEL_DIR IMG_DIR [RES_FILE]
Compute the iou of bounding boxes in XML_DIR and LABEL_DIR.
Store the results RES_FILE if it is provided.
EOF
if (@ARGV < 3 or @ARGV > 4) {
  print $usage;
  exit;
}

my $xml_dir = $ARGV[0];
my $label_dir = $ARGV[1];
my $img_dir = $ARGV[2];
my $res_file = $ARGV[3] if @ARGV == 4;
my $shift_file = $ARGV[4] if @ARGV == 5;
my @xml_files = glob "$xml_dir/*.xml";
my $xml_num = @xml_files;
my $count = 0;
my $bar_width = 70;
# image_name, iou, area_det, area_gt, xmin, ymin, xmax, ymax, xmin_gt, ymin_gt, xmax_gt, ymax_gt
my (@names, @shapes, @ious, @area_dets, @area_gts, @xmins, @ymins, @xmaxs, @ymaxs, @xmin_gts, @ymin_gts, @xmax_gts, @ymax_gts, @xshifts, @yshifts);

opendir XML_DIR, $xml_dir or die "Cannot open $xml_dir: $!";
foreach (readdir XML_DIR) {
  next unless /\.xml$/;
  my $xml = catfile($xml_dir, $_);
  s/\.xml/.txt/;
  my $label = catfile($label_dir, $_);
  s/\.txt/.jpg/;
  my $img = catfile($img_dir, $_);
  open XML, '<', $xml or die "Cannot open $xml: $!";
  my $xml_str = join "", <XML>;
  my $xmin = $1 if $xml_str =~ /<xmin>(.+)<\/xmin>/;
  my $ymin = $1 if $xml_str =~ /<ymin>(.+)<\/ymin>/;
  my $xmax = $1 if $xml_str =~ /<xmax>(.+)<\/xmax>/;
  my $ymax = $1 if $xml_str =~ /<ymax>(.+)<\/ymax>/;
  close XML;
  open LABEL, '<', $label or die "Cannot open $label: $!";
  my $label_str = join "", <LABEL>;
  my ($xmin_gt, $ymin_gt, $xmax_gt, $ymax_gt) = (split ' ', $label_str)[4, 5, 6, 7];
  close LABEL;
  my $shape_str = (split ' ', `identify '$img'`)[2];
  # print $img;

  my $area_det = &area($xmin, $ymin, $xmax, $ymax);
  my $area_gt = &area($xmin_gt, $ymin_gt, $xmax_gt, $ymax_gt);

  my $x_shift = 0;
  my $y_shift = 0;
  # if ($shape_str eq "1280x720") {
  #   $x_shift = 29;
  #   $y_shift = 29;
  # } else {
  #   $x_shift = 15;
  #   $y_shift = 15;
  # }
  # if ($area_det < 240) {
  #   $x_shift = 26;
  #   $y_shift = 26;
  # } else {
  #   $x_shift = 15;
  #   $y_shift = 15;
  # }

  $xmin = $xmin - $x_shift;
  $xmax = $xmax - $x_shift;
  $ymin = $ymin - $y_shift;
  $ymax = $ymax - $y_shift;

  my $xmin_inter = &max($xmin, $xmin_gt);
  my $ymin_inter = &max($ymin, $ymin_gt);
  my $xmax_inter = &min($xmax, $xmax_gt);
  my $ymax_inter = &min($ymax, $ymax_gt);
  my $area_inter = &area($xmin_inter, $ymin_inter, $xmax_inter, $ymax_inter);
  my $area_union = $area_det + $area_gt - $area_inter;
  my $iou = $area_inter / $area_union;

  push @names, $_;
  push @shapes, $shape_str;
  push @ious, sprintf("%.2f", $iou);
  push @area_dets, int($area_det);
  push @area_gts, int($area_gt);
  push @xmins, int($xmin);
  push @ymins, int($ymin);
  push @xmaxs, int($xmax);
  push @ymaxs, int($ymax);
  push @xmin_gts, int($xmin_gt);
  push @ymin_gts, int($ymin_gt);
  push @xmax_gts, int($xmax_gt);
  push @ymax_gts, int($ymax_gt);
  push @xshifts, int($xmin - $xmin_gt);
  push @yshifts, int($ymin - $ymin_gt);

  my $bar_count = ++$count / $xml_num * $bar_width;
  $| = 1;
  my $percent = sprintf("%.1f", $bar_count/$bar_width*100);
  print "\r[".("#"x$bar_count).(" "x($bar_width-$bar_count))."]".$percent."% ";
  # if ($_ eq "960184.txt") {
  #   print "\n";
  #   print "xml: $xml, label: $label\n";
  #   print "xmin = $xmin, ymin = $ymin, xmax = $xmax, ymax = $ymax\n";
  #   print "xmin_gt = $xmin_gt, ymin_gt = $ymin_gt, xmax_gt = $xmax_gt, ymax_gt = $ymax_gt\n";
  #   print "xmin_inter = $xmin_inter, ymin_inter = $ymin_inter, xmax_inter = $xmax_inter, ymax_inter = $ymax_inter\n";
  #   print "area_inter = $area_inter, area_union = $area_union\n";
  #   print "iou = $iou\n";
  #   exit;
  # } else {
  #   next;
  # }
}
print "\n";
closedir XML_DIR;

my $mean_iou = &sum(@ious) / @ious;
my $n_iou = @ious;
print "Total $n_iou records\n";
print "Mean IoU: $mean_iou\n";

if (defined $res_file) {
  open RES_FILE, '>', $res_file or die "Cannot open $res_file: $!";
  my @order = sort {$ious[$a] <=> $ious[$b]} 0..$#ious;
  foreach (@order) {
    print RES_FILE "$names[$_] $shapes[$_] $ious[$_] $area_dets[$_] $area_gts[$_] $xmins[$_] $ymins[$_] $xmaxs[$_] $ymaxs[$_] $xmin_gts[$_] $ymin_gts[$_] $xmax_gts[$_] $ymax_gts[$_] $xshifts[$_] $yshifts[$_]\n"
  }
  close RES_FILE;
}

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
