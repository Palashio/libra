function load(){
    var w = window.innerWidth; //window width includes area used for scroll bars so subtract some width later on
    if(w < 1300){
        w = 1300;
    }
    document.documentElement.style.setProperty("--screen-width", w); //code originally written on a 1920x969 monitor
    document.documentElement.style.setProperty("--screen-height", window.innerHeight);
    slideshow(0);
}

var prevSlide=5;

function slideshow(slide){
    if(prevSlide == slide){ //don't reanimate if the same tab was clicked consecutively
        return;
    }
    
//    graphic
    document.getElementById("g" + prevSlide).classList.remove("float");
    document.getElementById("g" + prevSlide).classList.add("drop");
    
//    description
    document.getElementById("d" + prevSlide).classList.remove("slideIn");
    
//    tab
    document.getElementById("t" + prevSlide).classList.remove("active");
    
    prevSlide=slide;
    
//    graphic
    document.getElementById("g" + slide).classList.remove("drop");
    document.getElementById("g" + slide).classList.add("lift");
    
//    description
    document.getElementById("d" + slide).classList.add("slideIn");
    
//    tab
    document.getElementById("t" + slide).classList.add("active");
    
    setTimeout(() => {
        document.getElementById("g" + slide).classList.remove("lift");
        document.getElementById("g" + slide).classList.add("float");
    }, 500);
}