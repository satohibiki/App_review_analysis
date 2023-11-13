$(document).ready(function() {
  // モーダルのボタンがクリックされたときの処理
  $("button.openModal").on("click", function() {
      var targetModalId = $(this).data("target");
      $("#" + targetModalId).fadeIn();
  });

  // モーダル内の閉じるボタンがクリックされたときの処理
  $(".closeModal button").on("click", function() {
      $(this).closest(".modalArea").fadeOut();
  });

  //アコーディオンをクリックした時の動作
  $('.title').on('click', function() {//タイトル要素をクリックしたら
    var findElm = $(this).next(".box");//直後のアコーディオンを行うエリアを取得し
    $(findElm).slideToggle();//アコーディオンの上下動作
      
    if($(this).hasClass('close')){//タイトル要素にクラス名closeがあれば
      $(this).removeClass('close');//クラス名を除去し
    }else{//それ以外は
      $(this).addClass('close');//クラス名closeを付与
    }
  });
});