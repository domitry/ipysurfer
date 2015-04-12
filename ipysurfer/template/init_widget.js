(function(){
    if(!_.isUndefined(IPython)){
	IPython.notebook.kernel.comm_manager.register_target("IPysurfer", function(comm){
	    console.log("Comm message has come.");
	    window.IPysurfer = {
		"comm": comm
	    };
	});
    }
})();
