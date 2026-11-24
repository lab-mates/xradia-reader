// Open an xradia-archive .h5 file in Fiji with the length unit set to "um".
//
// Usually not needed: File > Import > HDF5/N5/Zarr/OME-NGFF already reads
// the unit from the archive.  This macro is for the older HDF5 importer.
//
// Fiji's HDF5 plugin reads the voxel size from the element_size_um attribute
// but always labels it "micrometer" (hard-coded in the plugin, nothing in
// the file can change that).  This macro loads /data and relabels the unit;
// ImageJ displays "um" as "µm".
//
// Install: copy this file into Fiji.app/plugins/ and restart Fiji; it then
// appears as Plugins > Open Xradia Archive.  Or use Plugins > Macros > Run...
// From another macro: runMacro("Open_Xradia_Archive.ijm", "/path/scan.h5");

path = getArgument();
if (path == "")
    path = File.openDialog("Open Xradia HDF5 archive");
run("Scriptable load HDF5...", "load=[" + path + "] datasetnames=/data nframes=1 nchannels=1");
Stack.setXUnit("um");
Stack.setYUnit("um");
Stack.setZUnit("um");
rename(File.getNameWithoutExtension(path));
